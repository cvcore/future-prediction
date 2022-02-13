import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from futurepred.model.dynamics.flow_modules import WarpingLayer

class FlowLoss(nn.Module):

    def __init__(self, max_gt_const=3e-3, reduction='mean'):
        super().__init__()

        self._max_gt_const = max_gt_const
        self._backwarp = WarpingLayer(normalize_flow=False)
        self._reduction = reduction

    def forward(self, input: Tensor, target: dict):
        """ Arguments:
            input: flow output of the network
            target: groundtruth flow dictionary, keys are
                'flow_fwd'
                'flow_bwd'
                a mask will be calculated based on the consistency loss and groundtruth labels
                with a loss > _max_gt_const will be discarded
            Assumes input is backward flow
        """
        flo_f, flo_b = target['flow_fwd'], target['flow_bwd']
        const_losses = torch.abs(self._backwarp(flo_f, flo_b) + flo_b) # ideally ~= 0 in each pixel
        mask_valid = (const_losses < self._max_gt_const).float()

        loss = F.l1_loss(input, flo_b, reduction='none')
        loss *= mask_valid

        if self._reduction == 'mean':
            return loss.mean()
        return loss
