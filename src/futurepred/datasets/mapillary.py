import torch.utils.data as data
from collections import namedtuple
from pathlib import Path
from PIL import Image

class MapillaryVistas(data.Dataset):

    def __init__(self, root, split='train', transforms=None):

        self.root = root

        assert split in ['train', 'test', 'val'] # follow naming from CityScapes
        split_dir = {'train': 'training', 'test': 'testing', 'val': 'validation'}[split]

        self.transforms = transforms

        image_dir = Path(root) / split_dir / 'images'
        self.images = list(image_dir.glob('*.jpg'))

        self.target_dir = Path(root) / split_dir / 'labels'

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        image_path = self.images[index]
        image = Image.open(str(image_path)).convert('RGB')

        target_path = self.target_dir / (image_path.stem + '.png')
        target = Image.open(target_path)

        data = {'image': image, 'semantic': target}

        if self.transforms:
            data = self.transforms(data)

        return data

    MapillaryClass = namedtuple('MapillaryClass', ['name', 'id', 'cityscapes_train_id', 'color'])

    classes = [
        MapillaryClass('road',                   13, 0,  (128,64,128)),
        MapillaryClass('lane marking - general', 24, 0,  (128,64,128)),
        MapillaryClass('manhole',                41, 0,  (128,64,128)),
        MapillaryClass('curb',                    2, 1,  (244,35,232)),
        MapillaryClass('sidewalk',               15, 1,  (244,35,232)),
        MapillaryClass('building',               17, 2,  (70,70,70)),
        MapillaryClass('wall',                    6, 3,  (102,102,156)),
        MapillaryClass('fence',                   3, 4,  (190,153,153)),
        MapillaryClass('pole',                   45, 5,  (153,153,153)),
        MapillaryClass('utility pole',           47, 5,  (153,153,153)),
        MapillaryClass('traffic light',          48, 6,  (250,170,30)),
        MapillaryClass('traffic sign (front)',   50, 7,  (220,220,0)),
        MapillaryClass('vegetation',             30, 8,  (107,143,35)),
        MapillaryClass('terrain',                29, 9,  (152,251,152)),
        MapillaryClass('sky',                    27, 10, (70,130,180)),
        MapillaryClass('person',                 19, 11, (220,20,60)),
        MapillaryClass('bicyclist',              20, 12, (255,0,0)),
        MapillaryClass('motorcyclist',           21, 12, (255,0,0)),
        MapillaryClass('other rider',            22, 12, (255,0,0)),
        MapillaryClass('car',                    55, 13, (0,0,142)),
        MapillaryClass('truck',                  61, 14, (0,0,170)),
        MapillaryClass('bus',                    54, 15, (0,60,100)),
        MapillaryClass('on rails',               58, 16, (0,80,100)),
        MapillaryClass('motorcycle',             57, 17, (0,0,230)),
        MapillaryClass('bicycle',                52, 18, (119,11,32))
    ]
