import numpy as np
import torch.utils.data as data
import torch
from torch import distributed
import torch.nn as nn

import efficientPS.models as models
from efficientPS.models.panoptic_depth import NETWORK_INPUTS
from efficientPS.models.panoptic_perception import PanopticPerceptionNet

from efficientPS.config import load_config, DEFAULTS as DEFAULT_CONFIGS

from efficientPS.data import ISSTransform, PerceptionDataset, iss_collate_fn, perception_collate_fn
from efficientPS.data.sampler import DistributedARBatchSampler

from efficientPS.modules.fpn import FPN, FPNBody
# from efficientPS.modules.heads import FPNMaskHead, RPNHead
from efficientPS.modules.heads import FPNSemanticHeadDPC as FPNSemanticHeadDPC, SemanticDepthHead, FPNMaskHead, RPNHead, MultiLossHead, OpticalFlowHeadRAFT

from efficientPS.algos.fpn import InstanceSegAlgoFPN, RPNAlgoFPN
from efficientPS.algos.instance_seg import PredictionGenerator as MskPredictionGenerator, InstanceSegLoss
from efficientPS.algos.rpn import AnchorMatcher, ProposalGenerator, RPNLoss
from efficientPS.algos.detection import PredictionGenerator as BbxPredictionGenerator, DetectionLoss, ProposalMatcher
from efficientPS.algos.semantic_seg import SemanticSegLoss, SemanticSegAlgo
from efficientPS.algos.optical_flow import OpticalFlowAlgoRAFT, OpticalFlowLossRAFT
from efficientPS.algos.panoptic_depth import PanopticDepthAlgo, PanopticDepthLoss
from efficientPS.algos.multi_loss import MultiLossAlgo

from efficientPS.utils import logging
from efficientPS.utils.meters import AverageMeter, ConfusionMatrixMeter, ConstantMeter
from efficientPS.utils.misc import config_to_string, scheduler_from_config, norm_act_from_config, freeze_params, \
    all_reduce_losses, NORM_LAYERS, OTHER_LAYERS, NON_LAYERS
# from efficientPS.utils.panoptic_try import panoptic_stats, PanopticPreprocessing
from efficientPS.utils.parallel import DistributedDataParallel
from efficientPS.utils.snapshot import save_snapshot, resume_from_snapshot, pre_train_from_snapshots

from futurepred.utils.logger import Logger
logger = Logger.default_logger()

import os.path


def log_debug(msg, *args, **kwargs):
    logging.get_logger().debug(msg, *args, **kwargs)


def log_info(msg, *args, **kwargs):
    logging.get_logger().info(msg, *args, **kwargs)


def make_config(config_file):
    log_info("Loading configuration from %s", config_file)
    conf = load_config(config_file, DEFAULT_CONFIGS["flow"])

    log_info("\n%s", config_to_string(conf))
    return conf


def make_fpn_body(config=None):

    if config is None:
        file_path = os.path.abspath(os.path.dirname(__file__))
        config = os.path.join(file_path, "efficientPS", "config", "perception_run80_4_city_full.ini")
    config = make_config(config)

    body_config = config["body"]
    fpn_config = config["fpn"]

    # BN + activation
    norm_act_static, norm_act_dynamic = norm_act_from_config(body_config)

    # Create backbone
    log_debug("Creating backbone model %s", body_config["body"])
    body_fn = models.__dict__["net_" + body_config["body"]]
    body_params = body_config.getstruct("body_params") if body_config.get("body_params") else {}
    body = body_fn(norm_act=norm_act_static, **body_params)
    if body_config.get("weights"):
        body.load_state_dict(torch.load(body_config["weights"], map_location="cpu"))

    # Freeze parameters
    for n, m in body.named_modules():
        for mod_id in range(1, body_config.getint("num_frozen") + 1):
            if ("mod%d" % mod_id) in n:
                freeze_params(m)

    body_channels = body_config.getstruct("out_channels")

    # Create FPN
    fpn_inputs = fpn_config.getstruct("inputs")

    if 'efficient' in body_config['body']:
        fpn = FPN([body.corresponding_channels[2], body.corresponding_channels[3], body.corresponding_channels[5],
                   body.corresponding_channels[-1]],
                  fpn_config.getint("out_channels"),
                  fpn_config.getint("extra_scales"),
                  norm_act_static,
                  fpn_config["interpolation"])
    else:
        fpn = FPN([body_channels[inp] for inp in fpn_inputs],
                  fpn_config.getint("out_channels"),
                  fpn_config.getint("extra_scales"),
                  norm_act_static,
                  fpn_config["interpolation"])
    body = FPNBody(body, fpn, fpn_inputs)

    return body


class FPNFusion(nn.Module):

    @staticmethod
    def _make_down_or_upsampling(in_channels, in_stride, out_stride):

        if in_stride < out_stride:
            # downsampling with 3x3 conv, stride 2
            ds_factor = out_stride // in_stride
            out_channels = in_channels * 2
            conv = []
            for _ in range(int(np.log2(ds_factor))):
                conv.extend([
                    nn.Conv2d(in_channels, out_channels, 3, 2, 1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                ])
                in_channels = out_channels
                out_channels *= 2
            conv = nn.Sequential(*conv)
        elif in_stride > out_stride:
            # bilinear upsample, followed by 1x1 conv
            us_factor = in_stride // out_stride
            out_channels = in_channels // 2
            conv = []
            for _ in range(int(np.log2(us_factor))):
                conv.extend([
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    nn.Conv2d(in_channels, out_channels, 1, 1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                ])
                in_channels = out_channels
                out_channels //= 2
            conv = nn.Sequential(*conv)
        else:
            # pass-through
            conv = nn.Identity()

        return conv

    def __init__(self, fusion_level, fpn_strides=None, fpn_channels=None):

        super().__init__()

        if fpn_strides is None:
            fpn_strides = [4, 8, 16, 32]
        if fpn_channels is None:
            fpn_channels = [256, 256, 256, 256]

        fpn_n_level = len(fpn_strides)
        assert 0 <= fusion_level < fpn_n_level

        fusion_convs = []
        fusion_strd = fpn_strides[fusion_level]
        for stride, channel in zip(fpn_strides, fpn_channels):
            fusion_convs.append(
                self._make_down_or_upsampling(channel, stride, fusion_strd)
            )
        self.fusion_convs = nn.ModuleList(fusion_convs)
        self.fpn_n_level = fpn_n_level

    def forward(self, input):

        assert len(input) == self.fpn_n_level, f"Input FPN level {len(input)} mismatch with \
            fpn_n_level {self.fpn_n_level}!"

        out = [conv(feat) for feat, conv in zip(input, self.fusion_convs)]
        out = torch.cat(out, dim=1)

        return out


class FusedFPN(nn.Sequential):

    def __init__(self, fusion_level=1, feature_key='res5', config_path=None):
        super().__init__(
            make_fpn_body(config_path),
            FPNFusion(fusion_level)
        )

    def forward(self, x):
        x = super().forward(x)
        return {'res5': x} # only for compatibility with panoptic-deeplab encoder


def build_default_model(model_path, config_path):
    """Build FPN body with default parameters, and load the weights. """

    logger.info(f"Loading FPN body config from {config_path}")
    model = FusedFPN(config_path=config_path)
    logger.info(f"Loading FPN body weights from {model_path}")
    state_dict = torch.load(model_path)['state_dict']['body']
    model[0].load_state_dict(state_dict)

    return model
