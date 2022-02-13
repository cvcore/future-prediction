import torch
import torch.nn as nn
from torch.nn import Parameter

import futurepred.config.forecast as cfg

class ConditionalBatchNorm2d(nn.Module):
    """ The class-conditional batch normalization in arxiv:1907.06571 Efficient Video Generation on Complex Datasets

        This module is an extension to the standard BatchNorm2d to enable conditioning with an extra tensor, which is
        achieved by a linear embedding to map the conditioning variable onto a (mu, sigma) tuple for each feature
        channel.

        Furthermore, this module can turn off reduction over the time dimension, as required by the paper, to allow information
        passing between time steps. For ease of use with Conv2D layers, this module still accepts 4D input, but with
        the size of first dimension being (batch * n_frames).

        Input:
            x: 4D-Tensor of shape (b x T) x C x H x W
            class_id: Tensor of shape b x dim_cond
        Output:
            4D-Tensor of shape (b x T) x C x H x W
    """

    def __init__(self, in_channel, dim_cond, n_frame):
        """ Arguments:
            in_channel: number of input features, as in b x in_channel x H x W
            dim_cond: dimension of conditioning vector (noise vector)
            n_frame: number of frames in input tensor i.e., T
            sep_time_dim: if True, separate time dimension to avoid reducing over it when calculating running statistics
        """

        super().__init__()

        self.in_channel = in_channel
        self.n_frame = n_frame

        sep_time_dim = cfg.GENERATOR.NORM_SEP_TIME_DIM
        self.sep_time_dim = sep_time_dim
        if sep_time_dim:
            n_feature = self.n_frame * self.in_channel
        else:
            n_feature = self.in_channel

        self.bn = nn.BatchNorm2d(n_feature, affine=False) # batchnorm without learnable parameters
        self.embed = nn.Linear(dim_cond, n_feature * 2)
        self.embed.weight.data[:, :n_feature].normal_(1, 0.02)
        self.embed.weight.data[:, n_feature:].zero_()

    def forward(self, x, class_id):
        if self.sep_time_dim:
            # move time dimension and concatenate it along feature dimension
            # to calculate a unique batchnorm parameter for each time step
            BT, C, H, W = x.shape
            b = BT // self.n_frame
            n_feature = self.n_frame * self.in_channel
            x = x.view(b, n_feature, H, W)
        else:
            n_feature = self.in_channel
            class_id = class_id.repeat_interleave(self.n_frame, dim=0)

        out = self.bn(x)
        embed = self.embed(class_id)
        gamma, beta = embed.chunk(2, 1)
        gamma = gamma.view(-1, n_feature, 1, 1)
        beta = beta.view(-1, n_feature, 1, 1)
        out = gamma * out + beta

        if self.sep_time_dim:
            out = out.view(BT, C, H, W)
        return out


if __name__ == "__main__":

    cn = ConditionalBatchNorm2d(3, 2, 4)
    x = torch.rand([8, 3, 64, 64])
    class_id = torch.rand([2, 2])
    y = cn(x, class_id)
    print(cn)
    print(x.size())
    print(y.size())
