import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from warnings import warn
from math import floor, ceil


class LocalContext(nn.Sequential):

    def __init__(self, in_channels, kernel_size):
        assert len(kernel_size) == 3, "Kernel size should have shape (kt, ks1, ks2), got {}".format(kernel_size)
        # if in_channels % 2 != 0:
            # warn("channels should be (optimally) divisible by 2, got {}".format(in_channels))

        n_inplanes = in_channels // 2
        out_channels = in_channels // 2
        self.out_channels = out_channels

        (kt, ks1, ks2) = kernel_size
        same_padding = ( # we need same padding for odd- and even-sized kernels
                         # so not using padding argument in Conv3d
            floor((ks2-1)/2), ceil((ks2-1)/2), # left right
            floor((ks1-1)/2), ceil((ks1-1)/2), # top bottom
            floor((kt-1)/2), ceil((kt-1)/2)    # front back
        )

        super(LocalContext, self).__init__(
            # feature compression
            nn.Conv3d(in_channels, n_inplanes, (1, 1, 1), bias=False),
            nn.BatchNorm3d(n_inplanes),
            nn.ReLU(inplace=True),
            # spatial & temporal convolution
            nn.ConstantPad3d(same_padding, 0),
            nn.Conv3d(n_inplanes, out_channels, kernel_size, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )


class GlobalContext(nn.Module):

    def __init__(self, in_channels, div_factor, k_t):
        super(GlobalContext, self).__init__()
        # if in_channels % 3 != 0:
        #     warn("Warning: channels should be (optimally) divisible by 3, got {}".format(in_channels))
        n_inplanes = in_channels // 3
        self.out_channels = n_inplanes
        self.div_factor = div_factor
        self.k_t = k_t
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, n_inplanes, (1, 1, 1), bias=False),
            nn.BatchNorm3d(n_inplanes),
            nn.ReLU(inplace=True)
        )


    def forward(self, x):
        (T, H, W) = x.shape[-3:] # [T x H x W]
        if T >= self.k_t:
            k_pooling = (self.k_t, H//self.div_factor, W//self.div_factor)
        else:
            k_pooling = (T, H//self.div_factor, W//self.div_factor)
        x = F.avg_pool3d(x, k_pooling)
        x = self.conv(x)
        x = F.interpolate(x, (T, H, W), mode='trilinear', align_corners=True)
        return x


class TemporalBlock(nn.Module):
    """The Separable Temporal Block ('Dynamics module') in arXiv:2003.06409v2"""

    def __init__(self, in_channels, out_channels, k_spatial, k_temporal):
        super(TemporalBlock, self).__init__()

        k_s = k_spatial
        k_t = k_temporal

        local_kernels = [(k_t, k_s, k_s), (k_t, 1, k_s), (k_t, k_s, 1), (1, k_s, k_s)]
        global_div_factors = [1, 2, 4]

        self.in_channels = in_channels
        self.out_channels = out_channels

        if in_channels != out_channels:
            self.res_con = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, (1, 1, 1), bias=False),
                nn.BatchNorm3d(out_channels)
            )
        else:
            self.res_con = nn.Identity()

        self.local_mod = nn.ModuleList()
        conv_in_channels = 0
        for kernel in local_kernels:
            mod = LocalContext(in_channels, kernel)
            self.local_mod.append(mod)
            conv_in_channels += mod.out_channels

        self.global_mod = nn.ModuleList()
        for factor in global_div_factors:
            mod = GlobalContext(in_channels, factor, k_t)
            self.global_mod.append(mod)
            conv_in_channels += mod.out_channels

        self.conv_final = nn.Sequential(
            nn.Conv3d(conv_in_channels, out_channels, (1, 1, 1), bias=False),
            nn.BatchNorm3d(out_channels)
        )

        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        out = [] # list of tensors b x C_out x T_out x H x W
        for mod in self.local_mod:
            out.append(mod(x))

        for mod in self.global_mod:
            out.append(mod(x))

        out = torch.cat(out, dim=1) # cat the feature channels
        out = self.conv_final(out)
        out += self.res_con(x)
        out = self.activation(out)

        return out


class ResidualConv3dBlock(nn.Module):

    def __init__(self, channels, kernel_size, stride, n_layers, n_skip_layers=2):
        super().__init__()

        same_padding = tuple(k//2 for k in kernel_size)
        self.conv_layers = nn.ModuleList()

        for _ in range(n_layers):
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv3d(channels, channels, kernel_size, stride, same_padding, bias=False),
                    nn.BatchNorm3d(channels)
                )
            )

        self.activation = nn.ReLU(inplace=True)

        assert n_layers % n_skip_layers == 0, 'n_layers should be divisible by n_skip_layers'
        self.n_skip_layers = n_skip_layers

    def forward(self, x):
        identity = x

        for layer, module in enumerate(self.conv_layers, start=1):
            x = module(x)
            if layer % self.n_skip_layers == 0:
                x += identity
                identity = x
            x = self.activation(x)

        return x


class Dynamics(nn.Module):

    def __init__(self, in_channels, k_spatial, n_frames, layer_channels, drop_last=True):
        """ Wrapper Class for Temporal Block Dynamics Module

            args:
                in_channels: number of input features
                k_spatial: size of the spatial kernel
                n_frames: number of frames to be processed by the dynamics module
                layer_channels: list of output channels per layer, with the length of list equal to n_frames-1
                drop_last: bool, default True. Whether we should drop the last frame after each
                    temporal block.

            module input shape:
                batch x channel x T x H x W

            module output shape:
                batch x Cout x H x W
        """
        super().__init__()

        self.in_channels = in_channels
        self.n_frames = n_frames

        self.temp_blocks = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.drop_last = drop_last

        temp_inch = in_channels

        assert isinstance(layer_channels, list)
        assert not self.drop_last or len(layer_channels) == n_frames-1
        for layer, n_channel in enumerate(layer_channels):
            self.temp_blocks.append(TemporalBlock(temp_inch, n_channel, k_spatial, k_temporal=2))
            temp_inch = n_channel
            if layer != len(layer_channels)-1:
                self.residual_convs.append(ResidualConv3dBlock(temp_inch, (1,3,3), stride=1, n_layers=4, n_skip_layers=2))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        T = x.shape[2]
        assert T == self.n_frames, "Invalid input seqence length: {}".format(T)

        for mod_idx, mod in enumerate(self.temp_blocks):
            x = mod(x)
            if mod_idx != len(self.temp_blocks)-1:
                x = self.residual_convs[mod_idx](x) # 3x3 residual convolution between layers
            if self.drop_last:
                x = x[:, :, :-1, :, :] # drop last frame

        assert x.shape[2] == 1 or not self.drop_last
        return x[:, :, 0, :, :]
