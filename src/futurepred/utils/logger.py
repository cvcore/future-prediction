import functools
import logging
import sys
import warnings

import imgaug as ia
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from futurepred.datasets.cityscapes import Cityscapes
from fvcore.common.file_io import PathManager
from imgaug.augmentables.heatmaps import HeatmapsOnImage
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from termcolor import colored


class _ColorfulFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        self._root_name = kwargs.pop("root_name") + "."
        self._abbrev_name = kwargs.pop("abbrev_name", "")
        if len(self._abbrev_name):
            self._abbrev_name = self._abbrev_name + "."
        super(_ColorfulFormatter, self).__init__(*args, **kwargs)

    def formatMessage(self, record):
        record.name = record.name.replace(self._root_name, self._abbrev_name)
        log = super(_ColorfulFormatter, self).formatMessage(record)
        if record.levelno == logging.WARNING:
            prefix = colored("WARNING", "red", attrs=["blink"])
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            prefix = colored("ERROR", "red", attrs=["blink", "underline"])
        else:
            return log
        return prefix + " " + log

# cache the opened file object, so that different calls to `setup_logger`
# with the same file name can safely write to the same file.
@functools.lru_cache(maxsize=None)
def _cached_log_stream(filename):
    return open(filename, "a")

@functools.lru_cache()  # so that calling setup_logger multiple times won't add many handlers
def _setup_logger(
    output=None, distributed_rank=0, *, color=True, name="segmentation", abbrev_name=None
):
    """
    Initialize the segmentation logger and set its verbosity level to "INFO".
    Args:
        output (str): a file name or a directory to save log. If None, will not save log file.
            If ends with ".txt" or ".log", assumed to be a file name.
            Otherwise, logs will be saved to `output/log.txt`.
        name (str): the root module name of this logger
        abbrev_name (str): an abbreviation of the module, to avoid long names in logs.
            Set to "" to not log the root module in logs.
            By default, will abbreviate "segmentation" to "seg" and leave other
            modules unchanged.
    Returns:
        logging.Logger: a logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if abbrev_name is None:
        abbrev_name = "seg" if name == "segmentation" else name

    plain_formatter = logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%m/%d %H:%M:%S"
    )
    # stdout logging: master only
    if distributed_rank == 0:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        if color:
            formatter = _ColorfulFormatter(
                colored("[%(asctime)s %(name)s]: ", "green") + "%(message)s",
                datefmt="%m/%d %H:%M:%S",
                root_name=name,
                abbrev_name=str(abbrev_name),
            )
        else:
            formatter = plain_formatter
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # file logging: all workers
    if output is not None:
        if output.endswith(".txt") or output.endswith(".log"):
            filename = output
        else:
            filename = os.path.join(output, "log.txt")
        if distributed_rank > 0:
            filename = filename + ".rank{}".format(distributed_rank)
        # os.makedirs(os.path.dirname(filename))
        PathManager.mkdirs(os.path.dirname(filename))

        fh = logging.StreamHandler(_cached_log_stream(filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(plain_formatter)
        logger.addHandler(fh)

    return logger


class Logger(object):
    """ Helper function to help log losses into wandb, and text messages info console """

    def __init__(self, name='train', mute=False, video_fps=4, use_log_depth=True):
        """
        :param name: name of the logger
        :param mute: if True, all subsequent calls to this logger will be ignored - useful for distributed training such that only master node will print to console.
        """

        self.step = 0
        self.name = name
        self.mute_ = mute
        self.text_logger = _setup_logger(name=name)
        self.video_fps = video_fps
        self.use_log_depth = use_log_depth
        self.blender_frames = 0
        self.blender_input_image = None

        self.prediction_canvas = []

    @staticmethod
    def _segmentation_mask_image(image, logits, gt, caption='Segmentation'):
        assert logits.shape[-2:] == gt.shape[-2:], "Logits and groundtruth don't agree in shape!"
        h, w = logits.shape[-2:]

        mask_data_p = logits.argmax(dim=0).cpu().detach().numpy()
        mask_data_gt = gt.squeeze().cpu().detach().numpy()
        class_labels = {cls.train_id: cls.name
                        for cls in Cityscapes.classes if (cls.train_id!= -1 and cls.train_id != 255)}
        image = F.interpolate(image.unsqueeze(0), size=mask_data_gt.shape)

        mask_image = wandb.Image(image.squeeze().permute(1,2,0).detach().numpy(),
            caption=caption,
            masks={
                "predictions": {
                    "mask_data": mask_data_p,
                    "class_labels": class_labels
                },
                "ground_truth": {
                    "mask_data": mask_data_gt,
                    "class_labels": class_labels
                }
            })

        return mask_image

    @staticmethod
    def _segmentation_to_numpy(logits_or_mask, image=None):
        """
        convert segmentation mask into numpy array, colors are taken from cityscapes.
        :param logits_or_mask: tensor of shape C x H x W (logits) or 1 x H x W or H x W(mask)
        :param image: None or Tensor of shape 3 x H x W. image to be used as background.
        :return numpy array: shape 3 x H x W
        """
        assert len(logits_or_mask.shape) == 3 or len(logits_or_mask.shape) == 2, 'wrong input shape!'
        if len(logits_or_mask.shape) == 2:
            logits_or_mask = logits_or_mask.unsqueeze(0)

        if logits_or_mask.shape[0] > 1:
            mask = torch.argmax(logits_or_mask, dim=0, keepdim=True).cpu().detach().numpy()
        else:
            mask = logits_or_mask.cpu().detach().numpy()
        mask = mask.transpose((1,2,0))
        # mask[mask==255] = 40 # dirty hack to move background class to 40

        image_shape = (*mask.shape[:2], 3)
        if image is not None:
            image = image.cpu().detach().numpy().transpose((1,2,0))
            assert image_shape == image.shape, 'Image shape and mask shape doesn\'t agree!'
        else:
            image = np.zeros(image_shape)

        segmap = SegmentationMapsOnImage(mask.astype(np.int32), shape=image_shape)
        # if image is not None:
        #     out_image = segmap.draw_on_image(image)[0]
        # else:
        #     out_image = segmap.draw(size=image_shape[:2])[0]
        out_image = segmap.draw_on_image(image.astype(np.uint8), draw_background=False, background_class_id=255, alpha=1, colors=Cityscapes.get_colors(use_train_id=True))[0]
        out_image = out_image.transpose((2,0,1))

        return out_image

    def _segmentation_mask_video(self, logits_or_mask, image=None, wandb_video=True):
        """
        convert input logit / mask sequence into video, with optional image as background.
        :param logits_or_mask: Tensor of shape T x C x H x W
        :param image: Tensor of shape T x 3 x H x W or None
        :param wandb_video: if True, return wandb_video; otherwise return stacked mask array in shape T x 3 x H x W
        return: wandb.Video for logging
        """
        np_imgs = []
        len_seq = logits_or_mask.shape[0]
        for t in range(len_seq):
            if image is None:
                np_imgs.append(Logger._segmentation_to_numpy(logits_or_mask[t,...]))
            else:
                np_imgs.append(Logger._segmentation_to_numpy(logits_or_mask[t,...], image[t,...]))
        np_imgs = np.stack(np_imgs, axis=0)

        if wandb_video:
            return wandb.Video(np_imgs, fps=self.video_fps, format="gif")

        return np_imgs


    def log_semantic(self, image, logits, gt, suffix=''):
        """ log an overlay image in wandb for semantic segmentation.
            expected dimensions:
                image: 3 x H x W, type: FloatTensor
                logits: C x H x W, type: FloatTensor
                gt: 1 x H x W, type: LongTensor
        """
        if self.mute_:
            return
        mask_image = self._segmentation_mask_image(image, logits, gt)

        self.log({'semantic_img': [mask_image]}, suffix=suffix)

    def log_prediction_seq(self, image_seq=None, sem_gt_seq=None, sem_logit_seq=None, vis_entropy=False, depth_seq=None, flow_seq=None, suffix='', commit=True):
        """ log a gif video combining all future prediction types given

            Arguments:
                image_seq (tensor): T x 3 x H x W
                sem_gt_seq (tensor): T x 1 x H x W
                sem_logit_seq (tensor): T x class x H x W
                vis_entropy (bool): if True, visualize entropy from predicted semantic logits
                depth_seq (tensor): T x 1 x H x W
                flow_seq (tensor): T x 2 x H x W
                suffix (str): suffix for data row
                sample_idx (int): sample index
                commit (bool): if True, the video will be uploaded to wandb each time calling this function
                               otherwise, the video will be stacked in the horizontal direction until
                               commit=True is set. Useful for logging multiple sampled future.

        """

        canvas = []

        if image_seq is not None:
            image_seq_np = image_seq.cpu().detach().numpy()
            img_max, img_min = image_seq_np.max(), image_seq_np.min()
            image_seq_np = ((image_seq_np - img_min) / (img_max - img_min) * 255).astype(np.uint8)
            canvas.append(image_seq_np)

        if sem_gt_seq is not None:
            sem_gt_seq_np = self._segmentation_mask_video(sem_gt_seq, wandb_video=False)
            canvas.append(sem_gt_seq_np)

        if sem_logit_seq is not None:
            sem_logit_seq_np = self._segmentation_mask_video(sem_logit_seq, wandb_video=False)
            canvas.append(sem_logit_seq_np)

        if vis_entropy:
            entropy_seq_np = self._segmentation_entropy_video(sem_logit_seq, wandb_video=False)
            canvas.append(entropy_seq_np)

        if depth_seq is not None:
            canvas.append(self._depth_video(depth_seq, wandb_video=False))

        if flow_seq is not None:
            canvas.append(self._flow_video(flow_seq, wandb_video=False))

        canvas = np.concatenate(canvas, axis=3) # each element should have shape T x 3 x H x W, concatenate them horizontally
        self.prediction_canvas.append(canvas)

        if commit:
            video = np.concatenate(self.prediction_canvas, axis=2) # concatenate vertically
            self.prediction_canvas = []
            gif = wandb.Video(video, fps=self.video_fps, format='gif')
            self.log({'prediction_seq': gif}, suffix=suffix)


    @staticmethod
    def _segmentation_entropy_image(logits, caption='Entropy', wandb_image=True):
        """ calculate entropy for semantic prediction logits
            E = sum(p_i * log(p_i))
            arguments:

                logits: C x H x W
            return:
                wandb Image of entropy heatmap
        """
        logits = logits.cpu().detach().float()
        log_p = F.log_softmax(logits, dim=0)
        p = F.softmax(logits, dim=0)
        entropy = torch.sum(log_p*p, 0).numpy()
        entropy_image = ia.HeatmapsOnImage(entropy, shape=entropy.shape, min_value=np.min(entropy)-1e-6, max_value=np.max(entropy)+1e-6).invert().draw()[0]

        if wandb_image:
            return wandb.Image(entropy_image, caption=caption)

        return entropy_image

    def _segmentation_entropy_video(self, logit_seq, wandb_video=True):
        """ generate entropy video from logit sequence
            argument:
            logit_seq (tensor): T x class x H x W
            wandb_video: if True, return wandb.Video; otherwise return stacked np array
        """
        len_seq = logit_seq.shape[0]
        images = []

        for t in range(len_seq):
            entropy_image = Logger._segmentation_entropy_image(logit_seq[t], wandb_image=False)
            images.append(entropy_image)
        images = np.stack(images, axis=0).transpose(0,3,1,2)

        if wandb_video:
            return wandb.Video(images, fps=self.video_fps, format='gif')

        return images

    def _depth_image(self, depth_tensor, caption='depth', wandb_image=True):
        depth = np.float32(depth_tensor.permute(1,2,0).cpu().detach().numpy())
        if self.use_log_depth:
            mask = np.isfinite(depth)
            depth = -depth
            depth[~mask] = depth[mask].min()
        else:
            mask = depth>0
            depth[mask] = -np.log(depth[mask]) # plotting in log of inverse depth
            depth[~mask] = depth[mask].min()   # assigning invalid pixels to minimum value
        heatmap = HeatmapsOnImage(depth, depth.shape, depth.min(), depth.max())
        heatmap_rgb = heatmap.draw(cmap='viridis')[0]

        if wandb_image:
            return wandb.Image(heatmap_rgb, caption=caption)
        return heatmap_rgb

    def _depth_video(self, depth_seq, wandb_video=True):
        images = [self._depth_image(depth_frame, wandb_image=False) for depth_frame in depth_seq]
        images = np.stack(images, axis=0).transpose(0,3,1,2)

        if wandb_video:
            return wandb.Video(images, fps=self.video_fps, format='gif')
        return images


    def log_depth(self, pred, gt, suffix=''):
        """ log the depth prediction and groundtruth label
            expected dimensions:
                pred: 1 x H x W
                gt: 1 x H x W
        """
        if self.mute_:
            return
        disp_gt = self._depth_image(gt, 'depth gt', wandb_image=True)
        disp_pred = self._depth_image(pred, 'depth pred', wandb_image=True)

        self.log({'depth_img': [disp_pred, disp_gt]}, suffix=suffix)

    @staticmethod
    def _flow_image(flow_tensor, caption='', wandb_image=True):
        """ converts a flow tensor of shape (2, H, W) into numpy image (H, W, 3) """
        flow_map_np = flow_tensor.detach().cpu().numpy()
        _, h, w = flow_map_np.shape
        flow_map_np[:,(flow_map_np[0] == 0) & (flow_map_np[1] == 0)] = float('nan')
        rgb_map = np.ones((3,h,w)).astype(np.float32)
        normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())
        rgb_map[0] += normalized_flow_map[0]
        rgb_map[1] -= 0.5*(normalized_flow_map[0] + normalized_flow_map[1])
        rgb_map[2] += normalized_flow_map[1]
        rgb_map = np.transpose(rgb_map, (1,2,0)).clip(0,1)

        if wandb_image:
            return wandb.Image(rgb_map, caption=caption)
        return rgb_map

    def _flow_video(self, flow_seq, wandb_video=True):
        images = [self._flow_image(flow_frame, wandb_image=False) for flow_frame in flow_seq]
        images = np.stack(images, axis=0).transpose(0,3,1,2)

        if wandb_video:
            return wandb.Video(images, fps=self.video_fps, format='gif')
        return images

    def log_flow(self, pred, gt, suffix=''):
        if self.mute_:
            return
        flow_gt_wb = self._flow_image(gt, 'Flow groundtruth')
        flow_pred_wb = self._flow_image(pred, 'Flow prediction')

        self.log({'flow_img': [flow_pred_wb, flow_gt_wb]}, suffix=suffix)

    @staticmethod
    def _weight_on_image(weight, image=None, wandb_image=True):
        """Visualize a C-dimensional weight tensor as a RGB heatmap.
            If `image` is given, the heatmap will be drawn on image.
            Otherwise return a standalone heatmap.

        Args:
            weight (Tensor): C x H_b x W_b, with C being the number of channels.
                In case of a multi-channel heatmap, each channel will be visualized
                separately and stacked together in the horizontal direction.
            image (Optional[Tensor]): 3 x H x W
            wandb_image (bool): if True, return a wandb.Image object,
                otherwise return a numpy.ndarray

        Return:
            wandb.Image: shape (HxC) x W x 3 (same as image, if given. Otherwise H_b x W_b)
        """
        if image is not None:
            image = image.cpu().permute(1, 2, 0).numpy()
            heatmap_size = image.shape[:2]
        else:
            heatmap_size = weight[-2:]

        weight = weight.cpu().detach().permute(1, 2, 0).numpy()
        heatmap = HeatmapsOnImage(weight, heatmap_size, weight.min(), weight.max())

        if image is not None:
            h_image = np.hstack(heatmap.draw_on_image(image, alpha=0.5))
        else:
            h_image = np.hstack(heatmap.draw())

        if wandb_image:
            return wandb.Image(h_image, caption='weights_heatmap')
        return h_image

    @staticmethod
    def _blender_weight_on_image(blender_weights, image, wandb_image=True):
        """Visualize blender weights as a single RGB heatmap

        Args:
            blender_weights (Tensor): (T+1) x H_b x W_b
            image (Optional[Tensor]): T x 3 x H x W
                when image dimensions (H, W) are not equal to blender dimensions, blender weights
                will be bilinearly interpolated to the image dimension.

        Return:
            wandb.Image, if wandb_image == True
                otherwise np.ndarray with shape H x (Wx(T+1)) x 3
        """
        image = torch.cat([
            torch.zeros(1, *image.shape[1:]), # feature (F2F)
            image.cpu().detach() # history (F2M)
            ], dim=0) # (T+1) x 3 x H x W

        # convert to uint8 image
        if image.max() - image.min() != 0:
            image = ((image - image.min()) / (image.max() - image.min()) * 255.).byte()
        else:
            image = image.byte()

        h_map = []
        for i in range(blender_weights.shape[0]):
            h_map.append(Logger._weight_on_image(blender_weights[i].unsqueeze(0), image[i], False))
        h_map = np.hstack(h_map)

        if wandb_image:
            return wandb.Image(h_map, caption='weights_heatmap')
        return h_map

    def log_blender_weights(self, blender_weights, image=None):
        """Visualize blender weights as image or video, depending on input shape

        Args:
            blender_weights (Tensor): [T_future x ] (T+1) x H_b x W_b
                here T_future is an optional dimension. When given, the result will visualized
                as a gif video containing T_future frames.
                T is the number of frames used for warping (same below).
            image (Optional[Tensor]): T x 3 x H x W
                when image dimensions (H, W) are not equal to blender dimensions, blender weights
                will be bilinearly interpolated to the image dimension.
        """

        if len(blender_weights.shape) == 4:
            frames = [
                self._blender_weight_on_image(b_weight, image, wandb_image=False)
                for b_weight in blender_weights
            ]
            frames = np.stack(frames, axis=0)
            frames = frames.transpose(0, 3, 1, 2) # wandb needs T x 3 x H x W
            video = wandb.Video(frames, fps=self.video_fps, format='gif')
            self.log({'blender_weights_seq': video})
        else:
            image = self._blender_weight_on_image(blender_weights, image, wandb_image=True)
            self.log({'blender_weights': [wandb.Image(image, caption='weights_heatmap')]})

    def log_blender_weights_hook(self):
        """Return a hook for visualizing the result from the first batch to be used by the \
            torch.nn.Module.register_forward_hook() method.

        Output:
            Callable[module, input, output] -> None: the hook function
        """
        hook = lambda module, input, output: \
            self.log_blender_weights(output[:self.blender_frames], self.blender_input_image)
            # log first batch with `blender_frames` indicating the number of future frames

        return hook

    def log(self, log_dict, suffix='', *args, **kwargs):
        """ calls wandb.log with split as suffix, and adds automatically global_step to the dictionary """
        if not dict or self.mute_:
            return
        suffix_str = f"_{self.name}"
        if suffix:
            suffix_str = f"_{suffix}{suffix_str}"
        log_dict = {k+suffix_str:v for k,v in log_dict.items()}
        log_dict[f"global_step_{self.name}"] = self.step

        wandb.log(log_dict, *args, **kwargs)

    def commit(self):
        """ commit logs of current iteration to wandb and increase logging steps """
        self.log({}, commit=True)
        self.step += 1

    ########## Text Logging ##########
    def info(self, msg, *args, **kwargs):
        if not self.mute_:
            self.text_logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        if not self.mute_:
            self.text_logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        if not self.mute_:
            self.text_logger.error(msg, *args, **kwargs)

    def exception(self, msg, *args, **kwargs):
        if not self.mute_:
            self.text_logger.exception(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        if not self.mute_:
            self.text_logger.critical(msg, *args, **kwargs)
    ####################

    def mute(self):
        self.mute_ = True

    def unmute(self):
        self.mute_ = False

    default_logger_ = None
    @classmethod
    def default_logger(cls):
        if not cls.default_logger_:
            Logger.default_logger_ = Logger(name='general')
        return cls.default_logger_
