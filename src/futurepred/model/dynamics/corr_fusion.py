
import torch
from torch import nn
from torchvision import ops

from correlation import Correlation
from ..shared.deformable_conv import DeformConvPack


class DefConv2d(DeformConvPack):

    def __init__(self, in_feature, out_feature, kernel_size):
        padding = kernel_size // 2
        super().__init__(in_feature, out_feature, kernel_size, stride=1, padding=padding)
        self.bn = nn.BatchNorm2d(in_feature)

    def forward(self, x):
        x = self.bn(x)
        x = nn.functional.relu(x, inplace=True)
        return super().forward(x)


class SimilarityEmbedding(nn.Module):

    def __init__(self, in_feature, out_feature, kernel_size):
        super().__init__()
        self.conv_emb = nn.Conv2d(in_feature, out_feature, kernel_size, stride=1, padding=(kernel_size//2))

    def forward(self, x):
        x = self.conv_emb(x)
        x = nn.functional.normalize(x, p=2, dim=1)
        return x


class CorrFusion(nn.Module):
    """
    The spatio-temporal correspondence module from arXiv: 2101.10777v1

    This module is composed of a correlation (cost volume) generator, which calculates a correlation score in a 9x9 neighborhood,
    a fusion module, which is mainly for reducing the input feature size, and finally multiple neck layers to generate the shared
    representation.
    All 2D convolutions are replaced with deformable convolutions.

    Args:
        x (Tensor): 5D tensor of concatenated features from previous video frames.
                    Shape: batch x in_feature x in_frame x H x W

    TODO: try also to DCNv2, which contains an extra modulation parameter

    """

    def __init__(self, in_feature, in_frame, out_feature, fusion_feature, embedding_feature=128, max_disp=4):
        super().__init__()
        self.fusion = DefConv2d(in_feature*in_frame, fusion_feature, 1)

        self.embedding = SimilarityEmbedding(in_feature, embedding_feature, 3)
        self.corr = Correlation(pad_size=max_disp,
                                kernel_size=1,
                                max_displacement=max_disp,
                                stride1=1,
                                stride2=1,
                                corr_multiply=1)
        n_corr_feature = (in_frame-1) * (2*max_disp + 1)**2

        n_feature_neck = n_corr_feature + fusion_feature
        self.shared_neck = nn.Sequential(DefConv2d(n_feature_neck, n_feature_neck, 3),
                                         DefConv2d(n_feature_neck, n_feature_neck, 3),
                                         DefConv2d(n_feature_neck, n_feature_neck, 3),
                                         DefConv2d(n_feature_neck, n_feature_neck, 3),
                                         DefConv2d(n_feature_neck, n_feature_neck, 3),
                                         DefConv2d(n_feature_neck, out_feature, 3)
                                         )

        self.in_feature = in_feature
        self.in_frame = in_frame

    def forward(self, x):
        B, C, T, H, W = x.shape
        fusion_feat = self.fusion(x.reshape(B, -1, H, W))

        x_perm = x.permute(0, 2, 1, 3, 4).contiguous().view(-1, C, H, W)
        emb = self.embedding(x_perm).view(B, T, -1, H, W)
        emb_0 = emb[:, :-1, ...].reshape(B*(T-1), -1, H, W)
        emb_1 = emb[:, 1:, ...].reshape(B*(T-1), -1, H, W)
        c_vol = self.corr(emb_0, emb_1)
        c_vol = c_vol.view(B, -1, H, W)

        x = torch.cat([fusion_feat, c_vol], dim=1)
        x = self.shared_neck(x)

        return x
