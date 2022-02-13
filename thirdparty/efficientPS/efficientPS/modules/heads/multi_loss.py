import torch
import torch.nn as nn


class MultiLossHead(nn.Module):
    def __init__(self, loss_count=4):
        super(MultiLossHead, self).__init__()

        self.loss_count = loss_count
        self.loss_logvars = torch.nn.Parameter(torch.zeros(loss_count, requires_grad=True, dtype=torch.float32))

    # loss_trainable_list indicates if the weight of that loss is trainable.
    # If True, the weight is trainable. If False, the weight is not trainable.
    def forward(self, loss_list, loss_trainable_list):
        if loss_trainable_list[0]:
            factor = 0.5 * torch.exp(-self.loss_logvars[0])
            total_loss = (factor * loss_list[0]) + (0.5 * torch.log(1 + torch.exp(self.loss_logvars[0])))
        else:
            total_loss = loss_list[0]
        for i in range(1, len(loss_list)):
            if loss_trainable_list[i]:
                factor = 0.5 * torch.exp(-self.loss_logvars[i])
                total_loss = total_loss + (factor * loss_list[i]) + (0.5 * torch.log(1 + torch.exp(self.loss_logvars[i])))
            else:
                total_loss = total_loss + loss_list[i]

        return total_loss
