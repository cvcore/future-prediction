import torch
import torch.nn as nn
import torch.nn.functional as F

import random


class MultiLossAlgo:
    def __init__(self):
        pass

    def computeMultiLoss(self, multi_loss_head, loss_list, loss_trainable_list):
        # Run the semantic and instance heads here
        total_loss = multi_loss_head(loss_list, loss_trainable_list)

        loss_weights = torch.zeros_like(multi_loss_head.loss_logvars)
        for i in range(loss_weights.shape[0]):
            loss_weights[i] = torch.exp(-multi_loss_head.loss_logvars[i])

        return total_loss, loss_weights
