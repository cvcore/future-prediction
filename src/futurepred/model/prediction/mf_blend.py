import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F

from ..shared.deformable_conv import DeformConvPack

class DeformConvBlock(nn.Sequential):

    def __init__(self, in_feature, out_feature, n_layers=1, hidden_feature=None):
        assert n_layers == 1 or \
            n_layers > 1 and hidden_feature is not None

        layer = []
        n_feat_last = in_feature
        deform_conv = lambda in_f, out_f: DeformConvPack(
            n_feat_last, out_f, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False
        )

        for _ in range(n_layers-1):
            layer.append(deform_conv(n_feat_last, hidden_feature))
            layer.append(nn.BatchNorm2d(hidden_feature))
            layer.append(nn.ReLU())
            n_feat_last = hidden_feature
            hidden_feature = hidden_feature // 2
        layer.append(deform_conv(n_feat_last, out_feature))
        layer.append(nn.BatchNorm2d(out_feature))
        layer.append(nn.ReLU())

        super().__init__(*layer)


def backward_warp(feature, flow):
    """
    This function performs backward warping by sampling from `feature` w.r.t. `flow`

    Arguments:
        feature (Tensor [b x C x H x W])
        flow (Tensor [b x 2 x H x W])

    Output:
        Tensor (b x C x H x W): warped feature
    """

    if feature.shape[2] != flow.shape[2]:
        flow = F.interpolate(input=flow, scale_factor=feature.shape[2]/flow.shape[2], mode='bilinear')

    B, C, H, W = feature.size()

    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).to(feature.dtype).to(feature.device)
    vgrid = grid + flow

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W-1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H-1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)

    output = F.grid_sample(feature, vgrid, align_corners=False)

    return output


class MFBlend(nn.Module):
    """
    The F2MF module from Joint Forecasting of Features and Feature Motion for Dense Semantic Future Prediction (arXiv:2101.10777).

    This module predicts encoded feature in future timestep by using a F2M module, which predicts optical flow (motion) to warp past
    observations into future frames and a F2F module, which directly regresses a future feature (F) from past features.

    Arguments:
        in_feature (int): input features
        out_feature (int): output features
        n_dconv_layer (int): number of deformable convolution layers (default: 1, as in the paper)
        n_history (int): number of previous frames to consider for flow warping
        warp_direction (str): direction of warping. (default: backward)
            With 'backward' future features are backward warpped from previous features.
            With 'forward' we use splatting for forward-warping.
        out_intermediate (bool): if True, will return a dictionary containing F2M, F2F and the blended feature
            otherwise return directly the blended feature (default: True)

    Inputs for the forward function:
        input (dict): contains keys 'dynamics' and 'context'
            input['dynamics'] (Tensor[B x in_feature x H x W]): input dynamics feature
            input['context'] (Tensor[B x ctx_feature x T x H x W]): input context vectors, which are
                features encoded by the perception encoder for each previous frames. For blending, if
                `ctx_feature` is not equal to `out_feature`, it will be projected by an extra deform_conv
                layer.

    Output:
        If `out_intermediate` is set to True, output a dictionary with intermediate results
            containing `f2f` `f2m` and `f2mf`. Each one is a Tensor with shape [B x T x C x H x W]
        Otherwise output a single Tensor [B x T x C x H x W] with the final F2MF result

    """

    def __init__(self, in_feature, ctx_feature, out_feature, n_history, n_dconv_layer=1, warp_direction='backward', out_intermediate=True):
        super().__init__()

        self.f2m = DeformConvBlock(in_feature, 2, n_dconv_layer, in_feature//2)
        self.f2f = DeformConvBlock(in_feature, out_feature, n_dconv_layer, in_feature//2)
        self.bld_weights = nn.Sequential(
            DeformConvBlock(in_feature, 2, n_dconv_layer, in_feature//2),
            nn.Softmax(dim=1)
        )
        self.warp_direction = warp_direction
        self.out_intermediate = out_intermediate

        if ctx_feature != out_feature:
            self.project = nn.Sequential(
                DeformConvPack(ctx_feature, out_feature, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False),
                nn.BatchNorm2d(out_feature),
                nn.ReLU()
            )
        else:
            self.project = nn.Identity()


    def forward(self, input):

        dyn = input['dynamics']
        ctx = input['context']

        B, C, H, W = dyn.shape
        f2m_flow = self.f2m(dyn).reshape(-1, 2, H, W)

        Bc, Cc, Tc, Hc, Wc = ctx.shape
        ctx = ctx[:, :, -1, :, :]
        ctx = self.project(ctx)
        if self.warp_direction == 'backward':
            ctx_warp = backward_warp(ctx, f2m_flow)
        else:
            raise ValueError(f"Unsupported warping direction {self.warp_direction}!")
        ctx_warp = ctx_warp.unsqueeze(1)    # b x 1 x Cout x H x W

        feat = self.f2f(dyn).unsqueeze(1)   # b x 1 x Cout x H x W
        bld_w = self.bld_weights(dyn)       # b x 2 x H x W

        feat_s = torch.cat([feat, ctx_warp], dim=1)
        feat_s = feat_s.permute(2, 0, 1, 3, 4) # Cout x B x 2 x H x W

        bld_feat = (feat_s * bld_w).permute(2, 1, 0, 3, 4) # 2 x B x Cout x H x W
        out = bld_feat.mean(dim=0)

        if self.out_intermediate:
            f2f = bld_feat[0, ...]
            f2m = bld_feat[1:, ...].mean(dim=0)
            out = dict(
                blend = out,
                f2f = f2f,
                f2m = f2m,
                f2m_flow = f2m_flow
            )
        return out
