import argparse
import os
import torch
import yaml
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import pathlib
import math

from .maskflownet import config_folder as cf
from .maskflownet.model import MaskFlownet, MaskFlownet_S, Upsample, EpeLossWithMask



def normalize(img1, img2):
    rgb_mean = torch.cat((img1, img2), 2)
    rgb_mean = rgb_mean.view(rgb_mean.shape[0], 3, -1).mean(2)
    rgb_mean = rgb_mean.view(rgb_mean.shape[0], 3, 1, 1)

    return img1 - rgb_mean, img2-rgb_mean, rgb_mean


class FlowNet(nn.Module):

    def __init__(self, config_model="MaskFlownet.yaml", config_dataset="kitti.yaml", checkpoint="5adNov03-0005_1000000.pth", upsample=False):
        """ this is a meta function for importing the flow network
            args:
                config - model configuration file name
                checkpoint - model checkpoint file name
        """
        super(FlowNet, self).__init__()

        self.upsample = upsample

        mf_dir = pathlib.Path(__file__).parent.joinpath('maskflownet')
        config_model_path = mf_dir.joinpath('config_folder').joinpath(config_model)
        config_dataset_path = mf_dir.joinpath('config_folder').joinpath(config_dataset)
        weights_path = mf_dir.joinpath('weights').joinpath(checkpoint)

        with config_model_path.open() as f:
            config_model = cf.Reader(yaml.load(f))
        with config_dataset_path.open() as f:
            config_dataset = cf.Reader(yaml.load(f))

        state_dict = torch.load(weights_path)

        self.net = eval(config_model.value['network']['class'])(config_dataset)
        self.net.load_state_dict(state_dict)

        self.n_flow_feature = 529

    def forward(self, im0, im1):
        """ calculate optical flow from im0 and im1
            input shapes: [batch x C x H x W]
        """
        shape = im0.shape
        # resise image shape to be divisible by 64
        pad_h = (64 - shape[2] % 64) % 64
        pad_w = (64 - shape[3] % 64) % 64
        if pad_h != 0 or pad_w != 0:
            im0 = F.interpolate(
                im0, size=[shape[2] + pad_h, shape[3] + pad_w], mode='bilinear')
            im1 = F.interpolate(
                im1, size=[shape[2] + pad_h, shape[3] + pad_w], mode='bilinear')

        flows, masks, _, flow_feature = self.net(im0, im1)

        if not self.upsample:
            return {'flow': flows[-1], 'occ_mask': masks[0], 'feature': flow_feature}

        up_flow = Upsample(flows[-1], 4)
        up_occ_mask = Upsample(masks[0], 4)

        device = next(self.net.parameters()).device
        if pad_h != 0 or pad_w != 0:
            up_flow = F.interpolate(up_flow, size=[shape[2], shape[3]], mode='bilinear') * \
                torch.tensor([shape[d] / up_flow.shape[d]
                            for d in (2, 3)]).to(device).view(1, 2, 1, 1)
            up_occ_mask = F.interpolate(
                up_occ_mask, size=[shape[2], shape[3]], mode='bilinear')

        return {'flow': up_flow, 'occ_mask': up_occ_mask, 'feature': flow_feature}


class PWCNet(nn.Module):

    def __init__(self, feature_only=False, feature_level=2):
        """ Wrapper for the PWC-net
        Arguments
        feature_only (bool): if True, return only feature
        feature_level (int): valid ranges are 2-6. Representing from which decoder level to extract flow feature.
            the downsampling rates are 1/4 to 1/64 from level 2 to 6.
        """
        from .pytorch_pwc import run as pwc_run
        super().__init__()
        self.net = pwc_run.Network(feature_only, feature_level)
        self.n_flow_feature = 64 # extract feature from the decoder submodule, 2 layers before final flow regression
        self.feature_only = feature_only

    @torch.cuda.amp.autocast(False)
    def forward(self, im0, im1):

        assert im0.shape == im1.shape, 'Input images should have same dimention to calculate flow!'

        intHeight, intWidth = im0.shape[-2:]

        # tenPreprocessedFirst = im0.cuda().view(1, 3, intHeight, intWidth)
        # tenPreprocessedSecond = im1.cuda().view(1, 3, intHeight, intWidth)

        intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 64.0) * 64.0))
        intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 64.0) * 64.0))

        tenPreprocessedFirst = torch.nn.functional.interpolate(input=im0, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)
        tenPreprocessedSecond = torch.nn.functional.interpolate(input=im1, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)

        flowOut, flowFeature = self.net(tenPreprocessedFirst, tenPreprocessedSecond)
        if self.feature_only:
            return {'feature': flowFeature}

        tenFlow = 20.0 * torch.nn.functional.interpolate(input=flowOut, size=(intHeight, intWidth), mode='bilinear', align_corners=False)

        tenFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
        tenFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)

        return {'flow': tenFlow, 'feature': flowFeature}
