import numpy as np
import torch.utils.data as data
import torch
import torch.nn as nn
from torch import distributed
from torch.nn.parallel import DistributedDataParallel
import os

from efficientPS.utils.misc import norm_act_from_config
from efficientPS.modules.heads import FPNSemanticHeadDPC
from efficientPS.config import load_config, DEFAULTS as DEFAULT_CONFIGS



def make_config(config_file):
    conf = load_config(config_file, DEFAULT_CONFIGS["flow"])
    return conf


class FPNSeparator(nn.Module):

    @staticmethod
    def _make_down_or_upsampling(in_channels, in_stride, out_channels, out_stride):

        if in_stride < out_stride:
            # downsampling with 3x3 conv, stride 2
            ds_factor = out_stride // in_stride
            out_ch = in_channels * 2
            conv = []
            for _ in range(int(np.log2(ds_factor))):
                conv.extend([
                    nn.Conv2d(in_channels, out_ch, 3, 2, 1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True)
                ])
                in_channels = out_ch
                out_ch *= 2
            if in_channels != out_channels:
                conv.extend([
                    nn.Conv2d(in_channels, out_channels, 1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                ])
            conv = nn.Sequential(*conv)

        elif in_stride > out_stride:
            # bilinear upsample, followed by 1x1 conv
            us_factor = in_stride // out_stride
            out_ch = in_channels // 2
            conv = []
            for _ in range(int(np.log2(us_factor))):
                conv.extend([
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    nn.Conv2d(in_channels, out_ch, 1, 1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True)
                ])
                in_channels = out_ch
                out_ch //= 2
            if in_channels != out_channels:
                conv.extend([
                    nn.Conv2d(in_channels, out_channels, 1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                ])
            conv = nn.Sequential(*conv)

        else:
            # pass-through
            if in_channels != out_channels:
                conv = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            else:
                conv = nn.Identity()

        return conv

    def __init__(self, fusion_level, in_features, fpn_strides=None, fpn_channels=None):

        super().__init__()

        if fpn_strides is None:
            fpn_strides = [4, 8, 16, 32]
        if fpn_channels is None:
            fpn_channels = [64, 64, 64, 64]

        sep_mods = []
        for stride, channel in zip(fpn_strides, fpn_channels):
            sep_mods.append(
                self._make_down_or_upsampling(in_features, fpn_strides[fusion_level], channel, stride)
            )
        self.sep_mods = nn.ModuleList(sep_mods)

    def forward(self, input):

        out = [mod(input) for mod in self.sep_mods]
        return out


class FutureDecoderFusedFPN(nn.Module):

    def __init__(self, in_features, out_features, key_feature, config_path, fpn_channels=64, hidden_channels=32):

        super().__init__()

        config = make_config(config_path)

        sem_config = config['sem']
        body_config = config["body"]

        norm_act_static, _ = norm_act_from_config(body_config)

        self.fpn_head = FPNSemanticHeadDPC(
            fpn_channels,
            sem_config.getint("fpn_min_level"),
            sem_config.getint("fpn_levels"),
            out_features,
            hidden_channels=hidden_channels,
            pooling_size=sem_config.getstruct("pooling_size"),
            norm_act=norm_act_static
        )
        self.key_feature = key_feature
        channels = [fpn_channels, fpn_channels, fpn_channels, fpn_channels]
        self.sep_module = FPNSeparator(1, in_features, fpn_channels=channels)

    def forward(self, input):

        input = input[self.key_feature]
        input = self.sep_module(input)
        return self.fpn_head(input)[0]
