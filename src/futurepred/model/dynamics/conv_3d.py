### This file implements the Conv3D module in Chiu et al (arXiv:1904.10666v2).

import torch.nn as nn
import torch


def make_conv3d(in_channels, out_channels, kernel_size):
    return nn.Sequential(
        nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=(0, kernel_size[1]//2, kernel_size[2]//2),
            bias=False
        ),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True)
    )


class Dynamics(nn.Module):

    def __init__(self, in_channels, k_spatial, n_frames, layer_channels):

        super().__init__()

        conv_layers = []
        n_layers = len(layer_channels)
        assert n_layers == n_frames-1, f"n_frames={n_frames} mismatch with n_layers={n_layers}"
        last_channels = in_channels
        for layer_idx in range(n_layers):
            conv_layers.append(
                make_conv3d(last_channels, layer_channels[layer_idx], (2, k_spatial, k_spatial))
            )
            last_channels = layer_channels[layer_idx]

        self.conv_layers = nn.Sequential(*conv_layers)
        self.n_frames = n_frames

    def forward(self, x):

        assert self.n_frames == x.shape[2], \
            f"Got invalid frame length {x.shape[2]}, expect {self.n_frames}"

        x = self.conv_layers(x)
        x = x.squeeze(2)

        return x


if __name__ == "__main__":

    dyn = Dynamics(10, 3, 5, [80, 88, 96, 104])
    in_f = torch.zeros(2, 10, 5, 33, 65)
    out_f = dyn(in_f)
    print(out_f.shape)
