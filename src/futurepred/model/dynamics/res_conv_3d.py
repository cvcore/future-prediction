### This file implements the Conv3D module in Chiu et al (arXiv:1904.10666v2).

import torch.nn as nn
import torch


def make_conv3d(in_channels, out_channels, kernel_size, with_activation=True):

    return nn.Sequential(
        nn.ConstantPad2d(
            (kernel_size[2]//2, kernel_size[2]//2,
            kernel_size[1]//2, kernel_size[1]//2,
            0, kernel_size[1]//2),
            0.
        ),
        nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            bias=False
        ),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True) if with_activation else nn.Identity()
    )


class ResBlock3D(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size):

        super().__init__()

        self.res_block = nn.Sequential(
            make_conv3d(in_channels, out_channels, kernel_size, True),
            make_conv3d(out_channels, out_channels, kernel_size, False)
        )

        self.project = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, (1,1,1), bias=False),
            nn.BatchNorm3d(out_channels)
        )

    def forward(self, x):

        identity = x
        x = self.res_block(x)
        x += self.project(identity)
        x = nn.functional.relu(x, inplace=True)

        return x[:, :, 1:, :, :]


class Dynamics(nn.Module):

    def __init__(self, in_channels, k_spatial, n_frames, layer_channels):

        super().__init__()

        conv_layers = []
        n_layers = len(layer_channels)
        assert n_layers == n_frames-1, f"n_frames={n_frames} mismatch with n_layers={n_layers}"
        last_channels = in_channels
        for layer_idx in range(n_layers):
            conv_layers.append(
                ResBlock3D(last_channels, layer_channels[layer_idx], (2, k_spatial, k_spatial))
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
