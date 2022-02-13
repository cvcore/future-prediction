from .perception import PerceptionNetwork
from .encoder import PerceptionEncoder

import panoptic_deeplab

import futurepred.utils as utils
import torch

def build_backbone_from_cfg(cfg):

    backbone = cfg.TYPE
    backbone_path = cfg.PATH
    cfg_path = cfg.CFG

    if backbone == 'panoptic-deeplab':
        percep_model = panoptic_deeplab.build_default_model(model_path=backbone_path, cfg_path=cfg_path)
        backbone_out = percep_model.backbone
        if cfg.FREEZE:
            utils.model.freeze(backbone_out)
            backbone_out.eval()
        feature_dim = 2048
        feature_key = 'res5'
    elif backbone == 'resnet-custom':
        backbone_out = PerceptionEncoder()
        assert backbone_path is not None, "Saved model needed for resnet-custom, but got backbone_path==None"
        state_dict = torch.load(backbone_path, map_location=torch.device('cpu'))['state_dict']
        state_dict = utils.model.get_state_submodule(
            state_dict, 'module.encoder', remove_prefix=True)
        backbone_out.load_state_dict(state_dict, strict=True)
        if cfg.FREEZE:
            utils.model.freeze(backbone_out)
            backbone_out.eval()
        feature_dim = backbone_out.n_feature_out[-1]
        feature_key = 'res5'
    elif backbone == 'efficientps-fpn':
        # note this module only works with DistributedDataParallel!
        from . import efficient_ps
        backbone_out = efficient_ps.build_default_model(model_path=backbone_path, config_path=cfg_path)
        if cfg.FREEZE:
            utils.model.freeze(backbone_out[0])
            backbone_out[0].eval()
        feature_dim = 960
        feature_key = 'res5'
    else:
        raise Exception(f"Unsupported backbone: {backbone}")

    return backbone_out, feature_key, feature_dim
