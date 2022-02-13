import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init
# from ..registry import EXTRA_NECKS
# from mmdet.datasets.pipelines.flow_utils import vis_flow
from resample2d import Resample2d

from .tcea_fusion import TCEA_Fusion
from .conv_module import ConvModule
from .flow_modules import WarpingLayer, LiteFlowNetCorr
from .attention import CBAM


class BFPTcea(nn.Module):
    """BFP (Balanced Feature Pyrmamids)

    BFP takes multi-level features as inputs and gather them into a single one,
    then refine the gathered feature and scatter the refined results to
    multi-level features. This module is used in Libra R-CNN (CVPR 2019), see
    https://arxiv.org/pdf/1904.02701.pdf for details.

    Modified by Chengxin Wang for generating spatio-dynamics representations

    Args:
        in_channels (List[int]): Number of features for each input level, from bottom to top. For any input level with different
            number of features than ``refine_level``, an extra 1x1 projection convolution will be made during feature gathering.
        level_keys (List[str]): String keys for accessing feature in each level, from bottom to top.
        conv_cfg (dict): The config dict for convolution layers.
        norm_cfg (dict): The config dict for normalization layers.
        refine_level (int): Index of integration and refine level of BSF in multi-level features. Bottom feature has level 0.
        refine_type (str): Type of the refine op, currently support [None, 'conv', 'non_local'].
        n_frames:
        center: index of reference frame
        stack_type:
        conv_cfg:
        norm_cfg:
    """

    def __init__(self,
                 in_channels,
                 level_keys,
                 refine_level=1,
                 refine_type=None,
                 nframes=3,
                 center=None,
                 stack_type='add',
                 conv_cfg=None,
                 norm_cfg=None):
        super(BFPTcea, self).__init__()

        assert len(in_channels) == len(level_keys)
        self.num_levels = len(in_channels)
        self.in_channels = in_channels
        self.level_keys = level_keys
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.refine_level = refine_level
        self.refine_type = refine_type
        self.stack_type = stack_type

        self.nframes = nframes
        self.center = center
        assert 0 <= self.refine_level < self.num_levels

        self.in_project = []
        refine_channel = in_channels[refine_level]
        for level, channel in enumerate(in_channels):
            if channel != refine_channel:
                proj_conv = nn.Conv2d(channel, refine_channel, 1, 1, bias=True)
            else:
                proj_conv = nn.Identity()
            self.in_project.append(proj_conv)
        self.in_project = nn.ModuleList(self.in_project)

        bsf_channels = self.in_channels[refine_level]
        # liteflownet
        class Object():
            pass
        flow_args = Object()
        flow_args.search_range=4
        self.liteflownet = LiteFlowNetCorr(flow_args, bsf_channels+2)
        self.tcea_fusion = TCEA_Fusion(nf=bsf_channels,
                                       nframes=self.nframes,
                                       center=self.center)
        self.flow_warping = WarpingLayer()

        if self.refine_type == 'conv':
            self.refine = ConvModule(
                bsf_channels,
                bsf_channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg
                )
        elif self.refine_type == 'att':
            self.refine = nn.Sequential(
                ConvModule(
                    bsf_channels,
                    bsf_channels,
                    3,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg),
                CBAM(bsf_channels),
                )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def gather(self, inputs):
        # gather multi-level features by resize and average
        feats = []
        gather_size = inputs[self.level_keys[self.refine_level]].size()[2:]
        for i in range(self.num_levels):
            if i < self.refine_level:
                gathered = F.adaptive_max_pool2d(
                    self.in_project[i](inputs[self.level_keys[i]]), output_size=gather_size)
            else:
                gathered = F.interpolate(
                    self.in_project[i](inputs[self.level_keys[i]]), size=gather_size, mode='nearest')
            feats.append(gathered)
        bsf = sum(feats) / len(feats)
        return bsf

    def forward(self, inputs, ref_inputs, flow_init,
                next_inputs=None, next_flow_init=None):
        """Forward pass of BFPTcea module
            Arguments:
            inputs (dict[str, Tensor]): input feature pyramid
            ref_inputs (dict[str, Tensor]): reference feature pyramid
            flow_init:
        """
        assert len(inputs) == self.num_levels
        # inputs: B,C,H,W
        # Gather multi-level features by resize and average
        bsf = self.gather(inputs)
        ref_bsf = self.gather(ref_inputs)
        B,C,H,W = bsf.size()

        # assert flow_init.shape[-2:] == (H, W)
        flow_init = F.interpolate(flow_init, (H,W), mode='bilinear', align_corners=True)
        warp_bsf = self.flow_warping(ref_bsf, flow_init)
        flow_fine = self.liteflownet(bsf, warp_bsf, flow_init)
        warp_bsf = self.flow_warping(warp_bsf, flow_fine)

        if next_inputs is not None:
            next_bsf = self.gather(next_inputs)
            next_warp_bsf = self.flow_warping(next_bsf, next_flow_init)
            next_flow_fine = self.liteflownet(bsf, next_warp_bsf, next_flow_init)
            next_warp_bsf = self.flow_warping(next_warp_bsf, next_flow_fine)
            bsf_stack = torch.stack([warp_bsf, bsf, next_warp_bsf], dim=1)
            # B,3,C,H,W
        else:
            bsf_stack = torch.stack([bsf, warp_bsf], dim=1)
        bsf = self.tcea_fusion(bsf_stack)  # B,2,C,H,W -> B,1,C,H,W

        # Refinement
        if self.refine_type is not None:
            bsf = self.refine(bsf)

        # Scatter refined features to multi-levels by residual path
        # outs = []
        # for i in range(self.num_levels):
        #     out_size = inputs[i].size()[2:]
        #     if i < self.refine_level:
        #         residual = F.interpolate(bsf, size=out_size, mode='nearest')
        #     else:
        #         residual = F.adaptive_max_pool2d(bsf, output_size=out_size)
        #     outs.append(residual + inputs[i])
        # # return tuple(outs), flow_fine
        # return tuple(outs)

        return bsf
