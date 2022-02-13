from .temp_block import Dynamics as TempBlock
from .bfp_tcea import BFPTcea
from .corr_fusion import CorrFusion
from .temp_shift import DynamicsTempShift
from .conv_lstm import Dynamics as DynamicsConvLSTM
from .conv_3d import Dynamics as DynamicsConv3D
from .res_conv_3d import Dynamics as DynamicsResConv3D
from .s3d import Dynamics as DynamicsS3D
from .conv_gru import Dynamics as DynamicsConvGRU


def build_dynamics_model_from_cfg(cfg):

    model_map = {
        "TempBlock": TempBlock,
        "CorrFusion": CorrFusion,
        "TempShift": DynamicsTempShift,
        "ConvLSTM": DynamicsConvLSTM,
        "Conv3D":  DynamicsConv3D,
        "ResConv3D": DynamicsResConv3D,
        "SepInception": DynamicsS3D,
        "ConvGRU": DynamicsConvGRU
    }

    model_arguments = {
        "TempBlock": dict(
            in_channels=cfg.IN_CHANNELS,
            k_spatial=cfg.TB_K_SPATIAL,
            n_frames=cfg.N_FRAMES,
            layer_channels=cfg.TB_LAYER_CHANNELS,
            drop_last=cfg.TB_DROP_LAST
        ),
        "CorrFusion": dict(
            in_feature=cfg.IN_CHANNELS,
            in_frame=cfg.N_FRAMES,
            out_feature=cfg.OUT_CHANNELS,
            fusion_feature=cfg.N_FUSION,
        ),
        "TempShift": dict(
            in_channels=cfg.IN_CHANNELS,
            n_frames=cfg.N_FRAMES,
            n_div=cfg.TS_N_DIV,
            layer_channels=cfg.TS_LAYER_CHANNELS,
            use_non_local=cfg.TS_USE_NON_LOCAL
        ),
        "ConvLSTM": dict(
            in_channels=cfg.IN_CHANNELS,
            in_shape=cfg.CL_IN_SHAPE,
            k_spatial=cfg.CL_K_SPATIAL,
            layer_channels=cfg.CL_LAYER_CHANNELS
        ),
        "Conv3D": dict(
            in_channels=cfg.IN_CHANNELS,
            n_frames=cfg.N_FRAMES,
            k_spatial=cfg.C3D_K_SPATIAL,
            layer_channels=cfg.C3D_LAYER_CHANNELS
        ),
        "ResConv3D": dict(
            in_channels=cfg.IN_CHANNELS,
            n_frames=cfg.N_FRAMES,
            k_spatial=cfg.RC3D_K_SPATIAL,
            layer_channels=cfg.RC3D_LAYER_CHANNELS
        ),
        "SepInception": dict(
            in_channels=cfg.IN_CHANNELS,
            n_frames=cfg.N_FRAMES,
            layer_channels=cfg.S3D_LAYER_CHANNELS
        ),
        "ConvGRU": dict(
            in_channels=cfg.IN_CHANNELS,
            n_frames=cfg.N_FRAMES,
            k_spatial=cfg.CG_K_SPATIAL,
            layer_channels=cfg.CG_LAYER_CHANNELS
        )
    }

        # use_bfp_tcea = cfg.USE_BFP_TCEA
        # if use_bfp_tcea:
        #     assert use_flow, 'Warping BFP needs flow information. Please set use_flow=True!'
        #     self.bfp_module = BFPTcea([256, 512, 1024, 2048, 4096],
        #                               ['stem', 'res2', 'res3', 'res4', 'res5'],
        #                               refine_level=0,
        #                               refine_type='conv',
        #                               nframes=2,
        #                               center=1)
        #     self.bsf_project = nn.Conv2d(256, 4096, 1, 1, bias=True)
        # else:

    if cfg.TYPE not in model_map:
        raise ValueError(f"Unsupported dynamics module type: {cfg.TYPE}")

    return model_map[cfg.TYPE](**model_arguments[cfg.TYPE])
