import torch
import torch.nn as nn
from torchvision.models.resnet import Bottleneck

from .temp_shift_ops import TemporalShift
from .temp_non_local import NONLocalBlock3D

class _TempShiftBlock(Bottleneck):
    """ Extra stacked bottleneck layers with TSM inserted in the residual branch

        Input:
            Tensor[batch, channel, T, H, W]

        Output:
            Tensor[batch, channel, T, H, W]
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, n_frames, n_div):
        norm_layer = nn.BatchNorm2d
        downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * _TempShiftBlock.expansion, kernel_size=3, stride=1, padding=1),
            norm_layer(out_channels * _TempShiftBlock.expansion),
        )
        super().__init__(in_channels, out_channels, stride=1, downsample=downsample, norm_layer=nn.BatchNorm2d)
        self.n_frames = n_frames
        self.conv1 = TemporalShift(self.conv1, n_frames, n_div)

    def forward(self, x):
        b, c, t, h, w = x.shape
        assert self.n_frames == t, f"Got invalid number of input frames {t} in TempShiftBlock!"

        x = x.transpose(1, 2).reshape(b*t, c, h, w)
        x = super().forward(x)
        x = x.reshape(b, t, -1, h, w)
        x = x.transpose(1, 2)

        return x


class DynamicsTempShift(nn.Module):

    def __init__(self, in_channels, n_frames, n_div, layer_channels, use_non_local=None):
        """ Dynamics module with temporal shift block (arXiv:1811.08383)

            args:
                in_channels: number of input features
                n_frames: number of frames to be processed by the dynamics module
                n_div: number of chunks to split the features into. The first two chunks will be shifted
                    in temporal dimension
                layer_channels: list of output channels per layer, with the length of list equal to n_frames-1
                use_non_local (Optional[list[bool]]): optional list containing n_frames-1 bool elements,
                    each one denoting whether to use a non_local block after the corresponding layer.
                    By default, non-local block will be disabled for all layers.

            module input shape:
                batch x channel x T x H x W
        """
        super().__init__()
        assert n_frames > 1, "n_frames should > 1, got {}".format(n_frames)

        self.in_channels = in_channels
        self.n_frames = n_frames

        self.temp_blocks = nn.ModuleList()
        temp_inch = in_channels

        if use_non_local is None:
            use_non_local = [False] * (n_frames - 1)

        assert isinstance(layer_channels, list) and len(layer_channels) == n_frames-1
        assert isinstance(use_non_local, list) and len(use_non_local) == n_frames-1

        for layer, n_channel in enumerate(layer_channels):
            ts_block = _TempShiftBlock(temp_inch, n_channel, n_frames-layer, n_div)
            temp_inch = n_channel * ts_block.expansion
            if use_non_local[layer]:
                ts_block = nn.Sequential(
                    ts_block,
                    NONLocalBlock3D(temp_inch)
                )
            self.temp_blocks.append(ts_block)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        T = x.shape[2]
        assert T == self.n_frames, "Invalid input seqence length: {}".format(T)

        for block in self.temp_blocks:
            x = block(x)
            x = x[:, :, :-1, :, :] # drop last frame

        assert x.shape[2] == 1
        return x.squeeze(2)
