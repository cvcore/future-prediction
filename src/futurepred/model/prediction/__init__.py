from .decoder import FutureDecoderTransConv, FutureDecoderUpConv
from .distribution import ConditioningDistributions
from .generator import Generator
from .mf_blend import MFBlend
from .discriminator import SemanticsDiscriminator, DiscriminatorWrapper, SemanticsVideoDiscriminator
import math
import torch.nn as nn

def build_conditioner_from_cfg(cfg):

    cond_dist = ConditioningDistributions(
        latent_dim=cfg.DIM_LATENT,
        dist_dim=cfg.DIM_DIST,
        n_latent_past=cfg.N_LATENT_PAST,
        n_latent_future=cfg.N_LATENT_FUTURE
    )

    return cond_dist


def build_generator_from_cfg(cfg):

    future_generator = Generator(
        dynamics_dim=cfg.DIM_DYN,
        noise_dim=cfg.DIM_NOISE,
        layer_dim_base=cfg.DIM_LAYER_BASE,
        layer_dims=cfg.DIM_LAYER,
        n_future=cfg.N_FUTURE)

    return future_generator


def build_blender_from_cfg(cfg):
    if not cfg.ENABLE:
        return None

    blender = MFBlend(
        in_feature = cfg.IN_DIM,
        ctx_feature = cfg.CTX_DIM,
        out_feature = cfg.OUT_DIM,
        n_history = cfg.N_FRAME,
        n_dconv_layer = cfg.N_LAYER,
        warp_direction = cfg.WARP_DIR,
        out_intermediate = cfg.OUT_INTERMEDIATE
    )

    return blender


def build_future_decoder_from_cfg(cfg):

    model_dict = {
        'up-conv': FutureDecoderUpConv,
    }

    model_args = {
        'up-conv': dict(
            in_features=cfg.IN_DIM,
            key_feature=cfg.IN_KEY,
            out_features=cfg.OUT_DIM_SEMANTIC,
            odd_upsampling=cfg.ODD_UPSAMPLING
        )
    }

    if cfg.TYPE == 'fused-fpn':
        from .fpn_decoder import FutureDecoderFusedFPN
        model_dict['fused-fpn'] = FutureDecoderFusedFPN
        model_args['fused-fpn'] = dict(
            in_features=cfg.IN_DIM,
            out_features=cfg.OUT_DIM_SEMANTIC,
            key_feature=cfg.IN_KEY,
            fpn_channels=cfg.FPN_CHANNELS,
            config_path=cfg.FPN_CONFIG_PATH
        )

    semantic_head = model_dict[cfg.TYPE](**model_args[cfg.TYPE])

    return nn.ModuleDict({'semantic': semantic_head})


def build_discriminator_from_cfg(cfg, criterion):

    if not cfg.ENABLE:
        return None

    if cfg.TYPE == 'semantic':
        disc = SemanticsDiscriminator(cfg.N_CLASSES, cfg.N_LAYERS, cfg.N_FEATURES)
    elif cfg.TYPE == 'semantic_video':
        disc = SemanticsVideoDiscriminator(cfg.N_CLASSES)
    else:
        raise ValueError(f"Unsupported discriminator {cfg.TYPE}!")

    return DiscriminatorWrapper(disc, criterion)
