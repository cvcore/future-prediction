import cv2

import json
import os
from collections import namedtuple
import zipfile

from torchvision.datasets.utils import extract_archive, verify_str_arg, iterable_to_str
from torchvision.datasets.vision import VisionDataset
from PIL import Image

import torchvision.transforms.functional as TF
import torch
import numpy as np
import warnings


class Cityscapes(object):
    """`Cityscapes <http://www.cityscapes-dataset.com/>`_ Dataset. """

    CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])

    classes = [
        CityscapesClass('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('ego vehicle', 1, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('out of roi', 3, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('static', 4, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('dynamic', 5, 255, 'void', 0, False, True, (111, 74, 0)),
        CityscapesClass('ground', 6, 255, 'void', 0, False, True, (81, 0, 81)),
        CityscapesClass('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
        CityscapesClass('sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)),
        CityscapesClass('parking', 9, 255, 'flat', 1, False, True, (250, 170, 160)),
        CityscapesClass('rail track', 10, 255, 'flat', 1, False, True, (230, 150, 140)),
        CityscapesClass('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
        CityscapesClass('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
        CityscapesClass('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
        CityscapesClass('guard rail', 14, 255, 'construction', 2, False, True, (180, 165, 180)),
        CityscapesClass('bridge', 15, 255, 'construction', 2, False, True, (150, 100, 100)),
        CityscapesClass('tunnel', 16, 255, 'construction', 2, False, True, (150, 120, 90)),
        CityscapesClass('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
        CityscapesClass('polegroup', 18, 255, 'object', 3, False, True, (153, 153, 153)),
        CityscapesClass('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
        CityscapesClass('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
        CityscapesClass('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
        CityscapesClass('terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)),
        CityscapesClass('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),
        CityscapesClass('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
        CityscapesClass('rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),
        CityscapesClass('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
        CityscapesClass('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
        CityscapesClass('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
        CityscapesClass('caravan', 29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
        CityscapesClass('trailer', 30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
        CityscapesClass('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
        CityscapesClass('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
        CityscapesClass('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
        CityscapesClass('license plate', -1, -1, 'vehicle', 7, False, True, (0, 0, 142)),
    ]

    def __init__(self, root, split='train', mode='fine', target_types=['depth', 'semantic'], transforms=None):
        """ Args:
            root (string): Root directory of dataset where directory ``leftImg8bit``
                and ``gtFine`` or ``gtCoarse`` are located.
            split (string, optional): The image split to use, ``train``, ``test`` or ``val`` if mode="gtFine"
                otherwise ``train``, ``train_extra`` or ``val``
            mode (string, optional): The quality mode to use, ``gtFine`` or ``gtCoarse``
            target_type (string or list, optional): Type of target to use, ``instance``, ``semantic``, ``polygon``, ``depth``
                or ``color``. Can also be a list to output a tuple with all specified target types.
            transforms (callable, optional): A function/transform that takes input sample and its target as entry
                and returns a transformed version.
            sequence (tuple, optional): Tuple of (begin, end) denoting the range of the sequence. For example with (-3, 3) the output sequence will include timerange [-3, -2, -1, 0, 1, 2]
        """

        super().__init__()

        if isinstance(root, torch._six.string_classes):
            root = os.path.expanduser(root)
        self.root = root
        self.transforms = transforms

        assert mode in ['fine', 'coarse']
        self.mode = 'gtFine' if mode == 'fine' else 'gtCoarse'
        self.images_dir = os.path.join(self.root, 'leftImg8bit', split)
        self.targets_dir = os.path.join(self.root, self.mode, split)
        self.disparity_dir = os.path.join(self.root, 'disparity', split)
        self.camera_dir = os.path.join(self.root, 'camera', split)

        self.target_types = target_types
        self.split = split
        self.images = []
        self.targets = []

        verify_str_arg(mode, "mode", ("fine", "coarse"))
        if mode == "fine":
            valid_modes = ("train", "test", "val", "train_val")
        else:
            valid_modes = ("train", "train_extra", "val")
        msg = ("Unknown value '{}' for argument split with mode '{}'"
               "Valid values are {{{}}}.")
        msg = msg.format(split, mode, iterable_to_str(valid_modes))
        verify_str_arg(split, "split", valid_modes, msg)

        # [verify_str_arg(value, "target_types", ('semantic', 'depth', 'camera'))
            # for value in self.target_types]
        if 'depth' in self.target_types and 'camera' not in self.target_types: # calculating depth requires camera intrinsics
            self.target_types.append('camera')

        if not os.path.isdir(self.images_dir) or not os.path.isdir(self.targets_dir):
            raise RuntimeError('Dataset not found or incomplete. Please make sure all required folders for the'
                                   ' specified "split" and "mode" are inside the "root" directory')

        for city in os.listdir(self.images_dir):
            img_dir = os.path.join(self.images_dir, city)
            target_dir = os.path.join(self.targets_dir, city)
            disparity_dir = os.path.join(self.disparity_dir, city)
            camera_dir = os.path.join(self.camera_dir, city)
            for file_name in os.listdir(img_dir):
                curr_target = {}
                for t in self.target_types:
                    if t == "depth":
                        disparity_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0], 'disparity.png')
                        curr_target[t] = os.path.join(disparity_dir, disparity_name)
                    elif t == 'camera':
                        camera_name = f"{file_name.split('_leftImg8bit')[0]}_camera.json"
                        curr_target['camera'] = os.path.join(camera_dir, camera_name)
                    else:
                        target_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0],
                                                    self._get_target_suffix(self.mode, t))
                        curr_target[t] = os.path.join(target_dir, target_name)

                self.images.append(os.path.join(img_dir, file_name))
                self.targets.append(curr_target)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            a dictionary with keys: 'image' and targets like 'semantic', 'depth', if given.
        """

        # single image mode
        image = Image.open(self.images[index]).convert('RGB')

        targets = {}
        for target_type in self.target_types:
            if target_type in ['semantic', 'depth']:
                target = Image.open(self.targets[index][target_type])
            elif target_type == 'camera':
                with open(self.targets[index][target_type]) as json_file:
                    target = json.load(json_file)
            else:
                raise RuntimeError(f"Unsupported target type: {target_type}")
            targets[target_type] = target

        if self.transforms is not None:
            self.transforms.sample()
            trans_dict = self.transforms({'image': image, **targets})

        return trans_dict


    def __len__(self):
        return len(self.images)

    def extra_repr(self):
        lines = ["Split: {split}", "Mode: {mode}", "Type: {target_type}"]
        return '\n'.join(lines).format(**self.__dict__)

    def _load_json(self, path):
        with open(path, 'r') as file:
            data = json.load(file)
        return data

    def _get_target_suffix(self, mode, target_type):
        if target_type == 'instance':
            return '{}_instanceIds.png'.format(mode)
        elif target_type == 'semantic':
            return '{}_labelIds.png'.format(mode)
        elif target_type == 'color':
            return '{}_color.png'.format(mode)
        elif target_type == 'depth':
            return 'disparity.png'
        else:
            return '{}_polygons.json'.format(mode)

    @staticmethod
    def unique_train_ids():
        return np.unique([cls.train_id for cls in Cityscapes.classes])

    @staticmethod
    def train_id_to_name(train_id):
        if Cityscapes.train_id_to_name_ is None:
            # print('[Debug] creating train_id dict..')
            Cityscapes.train_id_to_name_ = {}
            for cls in Cityscapes.classes:
                if cls.train_id == 255 or cls.train_id == -1:
                    name = 'void'
                else:
                    name = cls.name
                Cityscapes.train_id_to_name_[cls.train_id] = name

        return Cityscapes.train_id_to_name_[train_id]

    @staticmethod
    def get_colors(use_train_id=True):
        """
        return a list of colors, each a tuple of int (R,G,B) for logging. The index corresponds to the id of each class.
        :param use_train_id: if True, usd `train_id` for each id. Otherwise `id` is used.
        """
        if use_train_id:
            if not Cityscapes.train_id_colors_:
                train_id_colors = [(0,0,0)]*256
                for cls in Cityscapes.classes:
                    if cls.train_id < 0 or cls.train_id > 255:
                        continue
                    train_id_colors[cls.train_id] = cls.color
                Cityscapes.train_id_colors_ = train_id_colors
            return Cityscapes.train_id_colors_

        if not Cityscapes.id_colors_:
            id_colors = [(0,0,0)]*256
            for cls in Cityscapes.classes:
                if cls.id < 0 or cls.id > 255:
                    continue
                id_colors[cls.id] = cls.color
            Cityscapes.id_colors_ = id_colors
        return Cityscapes.id_colors_

    @staticmethod
    def get_stuff_classes_train():
        if not Cityscapes.stuff_classes_train_:
            classes = []
            for cls in Cityscapes.classes:
                if cls.has_instances or cls.ignore_in_eval:
                    continue
                classes.append(cls.train_id)
            Cityscapes.stuff_classes_train_ = classes

        return Cityscapes.stuff_classes_train_


# static variables
Cityscapes.train_id_to_name_ = None
Cityscapes.id_colors_ = None
Cityscapes.train_id_colors_ = None
Cityscapes.stuff_classes_train_ = None
