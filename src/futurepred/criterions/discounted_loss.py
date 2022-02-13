import torch
import torch.nn as nn


class DiscountedLoss(nn.Module):
    """ Calculate weighted sum of losses along time axis.
        Assume input to be Tensor with shape [B, T, ...] where B, T stands for batch and time
        dimension.
        The final loss will be a weighted sum of individual losses in time dim, each having
        weight discount_factor^t, 0 <= t < T.
    """

    def __init__(self, criterion, discount_factor):
        super().__init__()

        self.criterion = criterion
        self.discount_factor = discount_factor
        self.dump_intermediate = False # debug

        assert discount_factor >= 0 and discount_factor <= 1, "Discount factor should be in range [0, 1], got {}".format(discount_factor)

    def forward(self, input, target):
        # todo: flatten the time dimension to speed up training?
        if not isinstance(target, dict):
            self._check_dim(input, target)

        loss = 0
        # for t in range(input.shape[1]):
        #     w = self.discount_factor**t
        #     if isinstance(target, dict): # for flow, which has 2 field as gt
        #         loss_cur = self.criterion(input[:, t, ...], {k: v[:, t, ...] for k, v in target.items()}) * w
        #     else:
        #         loss_cur = self.criterion(input[:, t, ...], target[:, t, ...]) * w
        #     loss += loss_cur
        #     if self.dump_intermediate:
        #         print('[Debug] t={}, loss={}'.format(t, loss_cur.item()))
        B, T = input.shape[:2]

        input = input.view(B*T, *input.shape[2:])
        target = target.view(B*T, *target.shape[2:])

        loss = self.criterion(input, target) # assume criterion doesn't come with any reduction

        if len(loss.shape) > 2:
            loss = torch.mean(loss, dim=list(range(1, len(loss.shape))))
        loss = loss.view(B, T)
        weights = torch.tensor([self.discount_factor]) ** torch.tensor(list(range(T)))
        weights = weights.to(loss)
        loss *= weights

        if self.dump_intermediate:
            print(f"[Debug] showing intermediate loss for batch 0")
            for t, l in enumerate(loss[0]):
                print(f"[Debug] loss(t={t}) = {l}")

        loss = loss.mean()

        return loss

    def _check_dim(self, x, y):
        assert x.shape[1] == y.shape[1], 'input and target tensor should have the same t dimension, got {} and {}'.format(x.shape[1], y.shape[1])
        assert x.shape[-1] == y.shape[-1], 'W dimension does not agree!'
        assert x.shape[-2] == y.shape[-2], 'H dimension does not agree!'
