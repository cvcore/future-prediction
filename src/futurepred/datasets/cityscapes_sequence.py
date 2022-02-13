import json
from pathlib import Path
from collections import namedtuple
import zipfile
import os

from torchvision.datasets.utils import extract_archive, verify_str_arg, iterable_to_str
from torchvision.datasets.vision import VisionDataset
from PIL import Image
import cv2

import torchvision.transforms.functional as TF
import torch
import numpy as np
import warnings

import umsgpack

from .cityscapes import Cityscapes

class CityscapesSequence(Cityscapes):
    """
    Dataset definition for the original Cityscapes dataset with support for sequential data (RGB and depth images).
    Groundtruth data without sequential labels will be returned as a single image, e.g. semantic, which corresponds to the 20th frame.
    """

    def __init__(self, root, split='train', mode='fine', target_types=['depth', 'semantic'], transforms=None, sequence_range=[0,30], expose_path=False):
        """
        Dataset for cityscape sequence.

        Arguments:
            root: refer to Cityscapes class
            split: refer to Cityscapes class
            mode: refer to Cityscapes class
            target_types: currently supports 'depth', 'semantic', 'semantic-eps', 'flow'
                'semantic' labels are generated with a pretrained resnet, and each label has (256, 512) resolution.
                'semantic-eps' labels are generated with EfficientPS with (1024, 2048) resolution
                'flow' labels are from RAFT, each value normalized to (-1, 1). Access them with 'flow_fwd' and 'flow_bwd' keys.
            transforms: a CombinedTransforms instance for image and target transformation
            sequence_range: specifies time range of each image sequence in the format of [start, end]. The index 'start' is inclusive, and 'end' exclusive.
            expose_path: when set to true, the path of the image sequence of each data item can be accessed under 'image_path'
        """
        super().__init__(root, split, mode, target_types, transforms)

        self.images_seq_dir = os.path.join(self.root, 'leftImg8bit_sequence', self.split)
        self.disparity_seq_dir = os.path.join(self.root, 'disparity_sequence', self.split)
        self.flow_seq_dir = os.path.join(self.root, 'flow_sequence', self.split)
        self.sequence_range = sequence_range
        self.expose_path = expose_path

        # semantic labels
        assert not ('semantic' in target_types and 'semantic-eps' in target_types), "You must at most pick one semantic label!"

        if 'semantic-eps' in target_types:
            self.semantic_seq_dir = os.path.join(self.root, 'gtPseudoSeqEPS', 'msk')
            md_path = Path(self.root) / 'gtPseudoSeqEPS' / 'metadata.bin'

            with md_path.open(mode='rb') as fd:
                metadata = umsgpack.unpack(fd, encoding='utf-8')
            id_to_cat = {}
            for img_meta in metadata['images']:
                id_to_cat[img_meta['id']] = img_meta['cat']
            assert len(id_to_cat) == 150000, 'Got incorrect number of mapping! Check metadata.bin!'
            self.id_to_cat = id_to_cat
        else:
            self.semantic_seq_dir = os.path.join(self.root, 'gtPseudoSeq', self.split)

        assert (len(sequence_range) == 2 and
            0 <= sequence_range[0] < 30 and
            0 < sequence_range[1] <= 30), \
            "Invalid sequence_range given: {}".format(sequence_range)


        self.images_seq = []

        for index, image_file in enumerate(self.images):
            file_name = os.path.basename(image_file)
            city, clip, seq, suffix = file_name.split('_')
            img_seq_dir = os.path.join(self.images_seq_dir, city)
            if 'depth' in target_types:
                disp_seq_dir = os.path.join(self.disparity_seq_dir, city)

            if 'semantic' in target_types:
                sem_seq_dir = os.path.join(self.semantic_seq_dir, city)
            elif 'semantic-eps' in target_types:
                sem_seq_dir = self.semantic_seq_dir

            if 'flow' in target_types:
                flow_seq_dir = os.path.join(self.flow_seq_dir, city)
            seq_head = int(seq) - 19

            # store samples into the following lists. They will later be accessed with index.
            image_seq = []
            disp_seq = []
            semantic_seq = []
            flow_seq = []
            for seq_idx in range(seq_head+sequence_range[0], seq_head+sequence_range[1]):
                image_name = '{}_{}_{:06d}_{}'.format(city, clip, seq_idx, suffix)
                image_seq.append(os.path.join(img_seq_dir, image_name))
                if "depth" in self.target_types:
                    disp_name = '{}_{}_{:06d}_{}'.format(city, clip, seq_idx, 'disparity.png')
                    disp_seq.append(os.path.join(disp_seq_dir, disp_name))
                if 'semantic' in self.target_types:
                    sem_name = '{}_{}_{:06d}_{}'.format(city, clip, seq_idx, 'gtPseudoSeq_labelTrainIds.png')
                    semantic_seq.append(os.path.join(sem_seq_dir, sem_name))
                if 'semantic-eps' in self.target_types:
                    sem_name = '{}_{}_{:06d}.png'.format(city, clip, seq_idx)
                    semantic_seq.append(os.path.join(sem_seq_dir, sem_name))
                if 'flow' in self.target_types and seq_idx-seq_head > 19:
                    flow_names = [
                        os.path.join(flow_seq_dir, f"{city}_{clip}_{seq_idx:06d}_flow_fwd.png"),
                        os.path.join(flow_seq_dir, f"{city}_{clip}_{seq_idx:06d}_flow_bwd.png")
                    ]
                    flow_seq.append(flow_names)

            self.images_seq.append(image_seq)
            if "depth" in self.target_types:
                self.targets[index]['depth_seq'] = disp_seq
            if 'semantic' in self.target_types or 'semantic-eps' in self.target_types:
                self.targets[index]['semantic_seq'] = semantic_seq
            if 'flow' in self.target_types:
                self.targets[index]['flow_seq'] = flow_seq

    def __len__(self):
        return len(self.images_seq)

    def __getitem__(self, index):
        """
        Arguments:
            index
        Return:
            A dictionary with the picked data
        """
        image_seq = [Image.open(image_path).convert('RGB') for image_path in self.images_seq[index]]

        # read files
        targets = {}
        for typ in self.target_types:
            if typ == "depth":
                targets['depth_seq'] = [Image.open(target_path) for target_path in self.targets[index]['depth_seq']]
                assert len(targets['depth_seq']) == len(image_seq), "Dataset error: depth sequence length and image sequence length are not the same!"
            elif typ == "semantic":
                targets['semantic_seq'] = [Image.open(target_path) for target_path in self.targets[index]['semantic_seq']]
            elif typ == "semantic-eps":
                semantic_seq = []
                for target_path in self.targets[index]['semantic_seq']:
                    base_name = target_path.rsplit('/', 1)[-1].split('.png')[0]
                    semantic_img = np.array(Image.open(target_path))
                    semantic_remapped = np.zeros_like(semantic_img)
                    for id, cat in enumerate(self.id_to_cat[base_name]):
                        semantic_remapped[semantic_img==id] = cat
                    semantic_img = semantic_remapped
                    semantic_img = Image.fromarray(semantic_img)
                    semantic_seq.append(semantic_img)
                targets['semantic_seq'] = semantic_seq
            elif typ == "camera":
                with open(self.targets[index][typ]) as json_file:
                    target = json.load(json_file)
                targets[typ] = target
            elif typ == "flow":
                fwd_flos = []
                bwd_flos = []

                for target_paths in self.targets[index]['flow_seq']:
                    def readflow(path):
                        flow_img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                        flow_img = (flow_img - 32768.) / 64.0 # unit after transformation: n. pixels
                        flow_img = flow_img[:, :, :2]
                        flow_shape = flow_img.shape
                        flow_img[:, :, 0] = flow_img[:, :, 0] / (flow_shape[0] - 1) # normalize v
                        flow_img[:, :, 1] = flow_img[:, :, 1] / (flow_shape[1] - 1) # normalize u
                        flow_img = torch.from_numpy(flow_img)
                        flow_img = flow_img.permute(2, 0, 1)
                        return flow_img

                    fwd_path, bwd_path = target_paths
                    fwd_flo = readflow(fwd_path)
                    bwd_flo = readflow(bwd_path)
                    fwd_flos.append(fwd_flo)
                    bwd_flos.append(bwd_flo)

                targets['flow_fwd'] = fwd_flos
                targets['flow_bwd'] = bwd_flos
            else:
                raise ValueError(f"Unsupported target type: {typ}")

        # per-frame transformation
        if self.transforms is not None:
            self.transforms.sample()

            for idx in range(len(image_seq)):
                input_dict = {'image': image_seq[idx]}

                for key, data in targets.items():
                    if key == 'depth_seq':
                        input_dict['depth'] = data[idx]
                    elif key == 'semantic_seq':
                        input_dict['semantic'] = data[idx]
                    elif key == 'camera':
                        continue
                    elif key in ['flow', 'flow_fwd', 'flow_bwd']:
                        if idx < len(data):
                            # dirty: flow is not aligned with targets, idx=0 starts at 20th frame
                            input_dict[key] = data[idx]
                    else:
                        raise RuntimeError(f"Unexpected key: {key}")

                trans_data = self.transforms(input_dict)

                for key, data in trans_data.items():
                    if key == 'depth':
                        targets['depth_seq'][idx] = data
                    elif key == 'semantic':
                        targets['semantic_seq'][idx] = data
                    elif key == 'camera':
                        continue
                    elif key in ['flow', 'flow_fwd', 'flow_bwd']:
                        if idx < len(targets[key]):
                            # dirty: flow is not aligned with targets, idx=0 starts at 20th frame
                            targets[key][idx] = data
                    elif key == 'image':
                        image_seq[idx] = data
                    else:
                        raise RuntimeError(f"Unexpected key: {key}")

        result =  {'image_seq': image_seq, **targets}
        if self.expose_path:
            result['image_path'] = self.images_seq[index]

        return result
