import torch
import torch.nn as nn
import torch.nn.functional as F
from futurepred.criterions import MultiTaskLoss

from . import encoder
from . import decoder
from .decoder import SingleFrameDecoder
import futurepred.utils
from .aspp import ASPP

class PerceptionNetwork(nn.Module):
    """ The perception network for single frame prediction, for this paper arXiv:2003.06409v2 it is used mainly for pretraining the shared encoder """
    DIM_PERCEPTION_OUT = 512

    def __init__(self, n_semantic_class, criterions={}, criterion_weights={}, use_mtl_uncertainty=False):
        """
        :param n_semantic_class: number of semantic classes to predict
        :param criterions: dictionaries of criterions to calculate the losses. Keys: 'semantic', 'depth'
        :param criterion_weights: weights of each criterion. Keys: 'semantic', 'depth'
        """
        super().__init__()

        self.encoder = encoder.PerceptionEncoder()
        n_encoder_feature = self.encoder.n_feature_out[-1]

        self.conv0 = None

        self.aspp = ASPP(n_encoder_feature, 256, (12,24,36))
        n_aspp_feature = 1280

        KEY_FEATURE = 'res5'
        N_BLOCKS = [3, 3, 3]
        DECODER_FEATURES = [256, 256, 256]
        USE_SKIP_CON = True
        SKIP_TYPE = 'cat'
        SKIP_FEATURES = [self.encoder.n_feature_out[1], self.encoder.n_feature_out[0], None]
        SKIP_FEATURES_PROJ = [64, 32, None]
        SKIP_KEYS = ['res2', 'stem', None]
        get_decoder = lambda n_feature, n_out_class, out_activation: SingleFrameDecoder(n_aspp_feature, n_out_class, KEY_FEATURE,
                                                                               n_blocks=N_BLOCKS,
                                                                               decoder_features=DECODER_FEATURES,
                                                                               use_skip_con=USE_SKIP_CON,
                                                                               skip_type=SKIP_TYPE,
                                                                               skip_features=SKIP_FEATURES,
                                                                               skip_features_proj=SKIP_FEATURES_PROJ,
                                                                               skip_keys=SKIP_KEYS,
                                                                               out_activation=out_activation)
        self.task_heads = nn.ModuleDict({'semantic': get_decoder(n_aspp_feature, n_semantic_class, nn.Identity()),
                                        #  'depth': get_decoder(n_aspp_feature, 1, nn.Identity())
                                        })

        self.criterions = criterions
        self.mtl = MultiTaskLoss(criterion_weights, use_mtl_uncertainty)


    def forward(self, input):
        image = input['image']

        in_shape = image.shape[-2:]
        in_batch = image.shape[0]
        image_curr = image

        out_encoder = self.encoder(image_curr)

        out_encoder['res5'] = self.aspp(out_encoder['res5'])

        loss_total = 0
        loss_tasks = {}
        out_tasks = {}

        for task_name, task_head in self.task_heads.items():
            task_pred = task_head(out_encoder)
            if task_pred.shape[-2:] != in_shape:
                task_pred = F.interpolate(task_pred, in_shape, mode='bilinear', align_corners=True)
            if task_name in self.criterions.keys():
                task_loss = self.criterions[task_name](task_pred, input[task_name])
                loss_tasks[task_name] = task_loss
            out_tasks[task_name] = task_pred.detach()

        loss_dict = self.mtl(loss_tasks)

        out_dict = dict(loss=loss_dict,
                    out_tasks=out_tasks)
        if self.mtl.use_mtl_uncertainty:
            out_dict['uncertainty'] = self.mtl.get_uncertainty()

        return out_dict
