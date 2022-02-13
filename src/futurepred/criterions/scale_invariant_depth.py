import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaleInvariantDepthLoss(nn.Module):
    """ Scale Invariant Depth Loss
        cf. http://arxiv.org/abs/1804.00607

        final loss = loss_{scale_invariant} + k_grad * loss_{grad}
    """

    def __init__(self, w_grad=0.5, n_grad_scale=4, input_logd=False, target_logd=False):
        """ arguments
            w_grad: weight of gradient matching term
            input_logd: if true, treat input as log depth value. Otherwise, treat it as absolute depth value.
            target_logd: see input_logd
        """
        super().__init__()
        self.w_grad = w_grad
        self.input_logd = input_logd
        self.target_logd = target_logd

    def forward(self, input, target):
        """ note for our network both input and target values are disparity """
        loss = self._scale_inv_loss(input, target)
        # disabled due to causing nans
        # ds_scales = [2, 4, 8, 16]
        # for ds in ds_scales:
        #     loss += self.w_grad * self._grad_matching_loss(input, target, ds)
        return loss

    def _scale_inv_loss(self, input, target):
        if self.target_logd:
            mask = torch.isfinite(target)
        else:
            mask = (target > 0) # only calculate the loss with valid groundtruth labels
        n_pixels = mask.sum()

        if not self.input_logd:
            assert (input[mask] > 0).all(), "Input contains negative depth value! Consider setting input_logd=True!"
            logd = torch.log(input[mask])
        else:
            logd = input[mask]

        if not self.target_logd:
            logd_gt = torch.log(target[mask])
        else:
            logd_gt = target[mask]

        residual = logd - logd_gt
        valid_pixels = torch.sum(mask)

        if valid_pixels > 0:
            loss_sinv = torch.sum(residual**2) / valid_pixels - (torch.sum(residual) / valid_pixels)**2
        else:
            loss_sinv = 0

        return loss_sinv

    def _grad_matching_loss(self, input, target, downsample=1):
        H, W = input.shape[-2:]
        assert H/downsample > 1 and W/downsample > 1, 'Input shape H,W = {},{} is too small for calculating gradient matching loss!'

        if downsample != 1:
            input = F.interpolate(input, scale_factor=1/downsample)
            target = F.interpolate(target, scale_factor=1/downsample)

        input = input.view(-1, *input.shape[-2:])
        target = target.view(-1, *target.shape[-2:])

        mask = (target != 0)
        n_pixels = mask.sum()

        logd = -torch.log(input)
        logd_gt = -torch.log(target)

        res = logd - logd_gt
        diff_h = res[:, 1:, :] - res[:, :-1, :]
        diff_w = res[:, :, 1:] - res[:, :, :-1]
        mask_h = (target[:, 1:, :]!=0) & (target[:, :-1, :]!=0)
        mask_w = (target[:, :, 1:]!=0) & (target[:, :, :-1]!=0)

        loss = torch.abs(diff_h[mask_h]).mean() + torch.abs(diff_w[mask_w]).mean()

        return loss


if __name__ == '__main__':
    from timeit import default_timer as timer

    input = torch.rand(50, 1, 32, 64)
    target = torch.rand(50, 1, 32, 64)
    crit = ScaleInvariantDepthLoss()
    start = timer()
    loss = crit(input, target)
    end = timer()
    print("Elapsed Time (CPU):", end-start)
    print(loss)

    input = input.cuda()
    target = target.cuda()
    start = timer()
    loss = crit(input, target)
    end = timer()
    print("Elapsed Time (GPU):", end-start)
    print(loss)
