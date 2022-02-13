import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import torch
import PIL
import warnings
import torch.nn.functional as F
import numpy as np
import random

from .cityscapes import Cityscapes
from .mapillary import MapillaryVistas

########## Helper ##########

def _make_locked_transform(transform):
    """ Class generator to be able to lock the random transforms
        the generated class supports lock(), unlock(), sample() and params() to control the random sampler for the transform.
    """
    class LockedTransform(transform):

        def __init__(self, *args, **kwargs):
            super(LockedTransform, self).__init__(*args, **kwargs)
            self.lock_ = False
            self.params_ = None

        def lock(self):
            """ Lock the random sampler, so each call to the transform object always has the same parameters. """
            self.lock_ = True

        def unlock(self):
            """ Unlock the random sampler, so each call always samples new parameters. """
            self.lock_ = False

        def sample(self):
            """ Sample once new set of parameters in the next transformation. Please note for unlocked transformation this method has no effect. """
            self.params_ = None

        def get_params(self, *args, **kwargs):
            if (not self.lock_) or (self.params_ is None):
                self.params_ = super(LockedTransform, self).get_params(*args, **kwargs)
            return self.params_

    return LockedTransform

########## Basic transformations ##########

def pad_image(image, size_base=32, bg_color=None):
    """ put tensor image of shape [..., C, H, W] into canvas [..., C, H_new, W_new] with background filled with fill_bg.
        This is mainly used for aligning the images during up / downsampling and especially useful during evaluation when the shape of input image is odd.
    """
    orig_shape = image.shape
    c, h, w = orig_shape[-3:]
    image = image.view(-1, c, h, w)
    h_new = (h + size_base - 1) // size_base * size_base + 1
    w_new = (w + size_base - 1) // size_base * size_base + 1

    canvas = torch.zeros((image.shape[0],c,h_new, w_new))
    assert bg_color is None or len(bg_color) == c
    if bg_color is not None:
        for i in range(c):
            canvas[:,i,:,:] = bg_color[i]
    canvas[:,:,:h,:w] = image
    canvas = canvas.reshape((*orig_shape[:-3], c, h_new, w_new))

    return canvas


def _pad_image_to(image, output_size, pad_value):
    """ pad the input image to output_size on the right and bottom borders """

    width, height = TF._get_image_size(image)
    height_out, width_out = output_size
    assert height_out >= height and width_out >= width, "output_size must be larger than input size!"

    padding = [0, 0, width_out - width, height_out - height]
    image = TF.pad(image, padding, pad_value, 'constant')

    return image


class _HorizontalFlip(torch.nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    @staticmethod
    def get_params(*args):
        return torch.rand(1)

    def forward(self, img, img_type=None):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        if self.get_params() < self.p:
            img = TF.hflip(img)
            if img_type is not None and img_type in ['flow', 'flow_fwd', 'flow_bwd']:
                # assume flows are tensor
                if not isinstance(img, torch.Tensor):
                    img = torch.from_numpy(np.array(img))
                # invert the u channel
                img[..., 0, :, :] *= -1
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class _RandomScale(torch.nn.Module):
    """ randomly scale the input PIL Image to a new shape """

    def __init__(self, scale_range, scale_step):
        super().__init__()

        assert len(scale_range) == 2, "scale_range must be a list of length 2!"
        self.scale_range = scale_range
        self.scale_step = scale_step

    @staticmethod
    def get_params(scale_range, scale_step):
        min_scale_factor, max_scale_factor = scale_range

        if min_scale_factor < 0 or min_scale_factor > max_scale_factor:
            raise ValueError('Unexpected value of min_scale_factor.')

        if min_scale_factor == max_scale_factor:
            return min_scale_factor

        # When step_size = 0, we sample the value uniformly from [min, max).
        if scale_step == 0:
            return random.uniform(min_scale_factor, max_scale_factor)

        # When step_size != 0, we randomly select one discrete value from [min, max].
        num_steps = int(round((max_scale_factor - min_scale_factor) / scale_step + 1))
        scale_factors = np.linspace(min_scale_factor, max_scale_factor, num_steps)
        np.random.shuffle(scale_factors)

        return scale_factors[0]

    def forward(self, img, img_type):
        new_scale = self.get_params(self.scale_range, self.scale_step)
        w, h = TF._get_image_size(img)
        new_h, new_w = int(h * new_scale), int(w * new_scale)

        if img_type == 'semantic':
            img = TF.resize(img, (new_h, new_w), PIL.Image.NEAREST)
        else:
            img = TF.resize(img, (new_h, new_w), PIL.Image.BILINEAR)

        return img

class _AdaptiveRandomCrop(transforms.RandomCrop):
    """ Crop the input to a given size. If the input is smaller, pad the boundary with constant number according to the input's type.
        This class supports the following inputs:
            'image' : RGB image, dim = 3
            'semantic' : Segmentation mask, dim = 1
            'flow' / 'flow_fwd' / 'flow_bwd' : Optical flow label, dim = 2
            'depth' : Depth label, dim = 1
    """

    def __init__(self, size, fill_map=None):
        """
        Arguments:
            size: Desired output size of the crop
            fill_map: Pixel fill value to use for each input type. If not given, the default mapping will be used.
        """
        super().__init__(size, pad_if_needed=True, padding_mode='constant')

        if fill_map:
            self.fill_map = fill_map
        else:
            self.fill_map = {'image': (0, 0, 0),
                             'semantic': 0,
                             'depth': 0,
                             'flow': 0, # flow are tensor, only support number (TF.pad)
                             'flow_fwd': 0,
                             'flow_bwd': 0}

    def forward(self, img, img_type):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.
            img_type: Input type

        Returns:
            PIL Image or Tensor: Cropped image.
        """
        if img_type in ['flow', 'flow_fwd', 'flow_bwd']:
            raise NotImplementedError("We don't support cropping normalized flow yet!")

        fill = self.fill_map[img_type]

        width, height = TF._get_image_size(img)
        # pad the width if needed
        if self.pad_if_needed and width < self.size[1]:
            padding = [self.size[1] - width, 0]
            img = TF.pad(img, padding, fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and height < self.size[0]:
            padding = [0, self.size[0] - height]
            img = TF.pad(img, padding, fill, self.padding_mode)

        i, j, h, w = self.get_params(img, self.size)

        return TF.crop(img, i, j, h, w)


class _CombinedTransforms(object):
    """ Combined transforms to transform groundtruth data of RGB image and segmentation mask together.
        This class takes care that when using random transforms, the same sampled parameters will be applied
        to the image and the targets.
    """

    def __init__(self, scale_range=None, crop_size=None, flip_prob=0,
                 image_resize_ratio=1, normalize_mean=None, normalize_std=None, image_pad_size=None):
        """
        arguments:
        :param crop_size: 2-Tuple: (h, w)
        :param scale_range: 2-Tuple: e.g. (0.8, 1.2) random scaling param or 3-Tuple with the last number as step for sampling
        :param flip_prob: prob of flipping. when flip_prob=0 never does the flip.
        :param image_resize_ratio: with >1, bilinear upsample the input image; and with <1 downsample
        :param normalize_mean: 3-Tuple. normalization factor for the RGB image
        :param normalize_std: 3-Tuple. normalization factor for the RGB image
        :param image_pad_size: 2-Tuple. if given, pad the image to this shape
        """
        super().__init__()

        self.transforms = [] # apply same transform as to image

        if scale_range is not None:
            # rand_scale = _make_locked_transform(transforms.RandomAffine)(degrees=0, scale=scale_range, resample=PIL.Image.BILINEAR)
            if len(scale_range) == 3:
                scale_step = scale_range[2]
                scale_range = scale_range[:2]
            else:
                scale_step = 0
            rand_scale = _make_locked_transform(_RandomScale)(scale_range, scale_step)
            rand_scale.lock()
            self.transforms.append(rand_scale)
        if crop_size is not None:
            if normalize_mean:
                mean_int = tuple([int(ch_mean*255) for ch_mean in normalize_mean])
                rand_crop = \
                    _make_locked_transform(_AdaptiveRandomCrop)(
                        crop_size,
                        fill_map={
                            'image': mean_int, # we perform normalization afterwards
                            'semantic': 0,
                            'flow': (0, 0),
                            'flow_fwd': (0, 0),
                            'flow_bwd': (0, 0),
                            'depth': 0}
                        )
            else:
                rand_crop = _make_locked_transform(_AdaptiveRandomCrop)(crop_size)
            rand_crop.lock()
            self.transforms.append(rand_crop)
        if flip_prob != 0:
            rand_flip = _make_locked_transform(_HorizontalFlip)(p=flip_prob)
            rand_flip.lock()
            self.transforms.append(rand_flip)

        self.target_transform = None
        self.image_resize_ratio = image_resize_ratio
        self.image_pad_size = image_pad_size

        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std

    def sample(self):
        """ generate new parameters for random transformations """

        for transform in self.transforms:
            transform.sample()

    def __call__(self, data_dict):
        """ arguments:
            data_dict: a dictionary of the data to be transformed.
                It contains keys of 'image', 'semantic', 'depth' and assumes that they have the same spatial shape.
        """

        for key, image in data_dict.items():

            # image transformation
            if key == 'image':
                img = data_dict['image']

                for transform in self.transforms:
                    if isinstance(transform, (_AdaptiveRandomCrop, _RandomScale, _HorizontalFlip)):
                        img = transform(img, key)
                    else:
                        img = transform(img)

                if self.image_resize_ratio != 1:
                    out_size = (int(img.size[-1]*self.image_resize_ratio), int(img.size[-2]*self.image_resize_ratio))
                    img = TF.resize(img, out_size)
                img = TF.to_tensor(np.array(img))

                if self.normalize_std is not None and self.normalize_mean is not None:
                    img = TF.normalize(img, self.normalize_mean, self.normalize_std)

                if self.image_pad_size is not None:
                    img = _pad_image_to(img, self.image_pad_size, 0)

                data_dict['image'] = img

            # target transformation
            elif key in ['semantic', 'depth', 'flow', 'flow_fwd', 'flow_bwd']:
                target = data_dict[key]

                for transform in self.transforms:
                    if isinstance(transform, (_AdaptiveRandomCrop, _RandomScale, _HorizontalFlip)):
                        target = transform(target, key)
                    else:
                        target = transform(target)

                if not isinstance(target, torch.Tensor):
                    if not isinstance(target, np.ndarray): # suppress warning
                        target = np.array(target)
                    if key == 'depth':
                        target = np.float32(target)
                    target = torch.from_numpy(target).unsqueeze(0)

                data_dict[key] = target

            else:
                raise ValueError(f"Unsupported key {key}")

        if self.target_transform:
            data_dict = self.target_transform(data_dict)

        return data_dict


class _SeqCombinedTransforms(_CombinedTransforms):
    """ The unwrapping in time-dimension is done in dataset,
        therefore here we use the same transform as single
        frame. """
    pass


class _CityscapesTargetTransform(object):
    """ Transformation for Cityscape target images (segmentation mask, depth) """

    def __init__(self, target_resize_ratio=1, use_train_ids=True, use_log_depth=True, target_pad_size=None):
        """
        args:
            target_resize_ratio: output_size / input_size, useful for
            use_train_ids: whether to use the train_ids for semantic mask
        """

        super().__init__()
        self.target_resize_ratio = target_resize_ratio
        self.use_train_ids = use_train_ids
        self.use_log_depth = use_log_depth
        self.target_pad_size = target_pad_size

    def __call__(self, targets):
        """
        args:
            targets: dictionary of target images. Keys should contain 'semantic', 'depth'
            use_log_depth: If true, preprocess depth into log space. Invalid depth measurement will be assigned to 'Inf'.
                           Otherwise, depth will be absolute value, with invalid pixels assigned to '0'
        """

        for typ, target in targets.items():
            if typ == "depth":
                fx = targets['camera']['intrinsic']['fx']
                baseline = targets['camera']['extrinsic']['baseline']
                mask = (target>0)
                target[mask] = (target[mask] - 1.) / 256.
                mask = (target>0)
                target[mask] = baseline * fx / target[mask]
                if self.use_log_depth:
                    target[mask] = torch.log(target[mask])
                    target[~mask] = float('-Inf')

            elif typ == 'semantic':
                target = target.long()
                if self.use_train_ids:
                    mapped = torch.zeros_like(target)
                    for cls in Cityscapes.classes:
                        mask = (target==cls.id)
                        mapped[mask] = cls.train_id
                    target = mapped
                    # in cityscapes.py, the training id of license plate remains -1, however we ignore it as well
                    target[target==-1] = 255

            elif typ in ['flow', 'flow_fwd', 'flow_bwd']:
                pass

            else:
                continue

            if self.target_resize_ratio != 1:
                target = target.unsqueeze(0)
                if typ == 'semantic':
                    target = target.float()
                    target = F.interpolate(target, scale_factor=self.target_resize_ratio, mode='nearest', recompute_scale_factor=False).squeeze(0).long() # use nearest to avoid corrupting class id
                else:
                    target = F.interpolate(target, scale_factor=self.target_resize_ratio, mode='bilinear', align_corners=False, recompute_scale_factor=False).squeeze(0)

            if self.target_pad_size:
                if typ == 'semantic':
                    target = _pad_image_to(target, self.target_pad_size, 255)
                elif typ == 'depth':
                    if self.use_log_depth:
                        target = _pad_image_to(target, self.target_pad_size, float('-Inf'))
                    else:
                        target = _pad_image_to(target, self.target_pad_size, 0)
                elif typ in ['flow', 'flow_fwd', 'flow_bwd']:
                    target = _pad_image_to(target, self.target_pad_size, 0) # use number as fill arg for tensor
                else:
                    raise ValueError(f"Unrecognized target type: {typ}")

            targets[typ] = target

        return targets


class _CityscapesSeqTargetTransform(_CityscapesTargetTransform):
    """ target transform for pseudo-gt data.
        This class has minor difference from its parent class:
            1. `use_train_id` in parent initialization will be locked to `false` because our gt data has already been transformed into `train_id`
    """
    def __init__(self, target_resize_ratio, use_log_depth, target_pad_size):
        super().__init__(target_resize_ratio=target_resize_ratio, use_train_ids=False, use_log_depth=use_log_depth, target_pad_size=target_pad_size)


class _MapillaryTargetTransform(object):

    def __init__(self, target_resize_ratio=1, convert_class_ids=True):
        self.target_resize_ratio = target_resize_ratio
        self.convert_class_ids = convert_class_ids

    def __call__(self, targets):
        """
        args:
            targets: dictionary of target images. Keys should contain 'semantic', 'depth'
        """

        for typ, target in targets.items():
            if typ == 'semantic':
                target = (target*255).long()
                if self.convert_class_ids:
                    mapped = torch.ones_like(target)*255
                    for cls in MapillaryVistas.classes:
                        mask = (target==cls.id)
                        mapped[mask] = cls.cityscapes_train_id
                    target = mapped
                    # in cityscapes.py, the training id of license plate remains -1, however we ignore it as well
            else:
                raise Exception("unsupported target type!")

            if self.target_resize_ratio != 1:
                target = target.unsqueeze(0)
                if typ == 'semantic':
                    target = target.float()
                    target = F.interpolate(target, scale_factor=self.target_resize_ratio, mode='nearest', recompute_scale_factor=False).squeeze(0).long() # use nearest to avoid corrupting class id
                else:
                    target = F.interpolate(target, scale_factor=self.target_resize_ratio, mode='bilinear', align_corners=False, recompute_scale_factor=False).squeeze(0)

            targets[typ] = target

        return targets


class CityscapesCombinedTransforms(_CombinedTransforms):

    def __init__(self,
                 scale_range=None, crop_size=None, flip_prob=0,
                 image_resize_ratio=1, target_resize_ratio=1, normalize_mean=None, normalize_std=None,
                 use_train_ids=False, use_log_depth=True,
                 image_pad_size=None, target_pad_size=None):

        super().__init__(scale_range, crop_size, flip_prob, image_resize_ratio, normalize_mean, normalize_std, image_pad_size)

        self.target_transform = _CityscapesTargetTransform(target_resize_ratio=target_resize_ratio,
                                                           use_train_ids=use_train_ids,
                                                           use_log_depth=use_log_depth,
                                                           target_pad_size=target_pad_size)


class CityscapesSeqCombinedTransforms(_SeqCombinedTransforms):
    def __init__(self,
                 flip_prob=0,
                 normalize_mean=None,
                 normalize_std=None,
                 scale_range=(0.25,0.25),
                 crop_size=(256,512),
                 image_resize_ratio=0.25,
                 target_resize_ratio=1,
                 use_log_depth=True,
                 image_pad_size=None,
                 target_pad_size=None):

        super().__init__(flip_prob=flip_prob,
                         normalize_mean=normalize_mean,
                         normalize_std=normalize_std,
                         scale_range=scale_range,
                         crop_size=crop_size,
                         image_resize_ratio=image_resize_ratio,
                         image_pad_size=image_pad_size)

        self.target_transform = _CityscapesSeqTargetTransform(target_resize_ratio=target_resize_ratio,
                                                              use_log_depth=use_log_depth,
                                                              target_pad_size=target_pad_size)


class MapillaryCombinedTransforms(_CombinedTransforms):

    def __init__(self, scale_range=None, crop_size=None, flip_prob=0, image_resize_ratio=1, target_resize_ratio=1, normalize_mean=None, normalize_std=None, convert_class_ids=True):
        super().__init__(scale_range, crop_size, flip_prob, image_resize_ratio, normalize_mean, normalize_std)
        self.target_transform = _MapillaryTargetTransform(target_resize_ratio=target_resize_ratio, convert_class_ids=convert_class_ids)
