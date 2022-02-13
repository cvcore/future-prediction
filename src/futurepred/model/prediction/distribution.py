import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
from torch.distributions import Normal, Independent

class DiagonalGaussian(nn.Module):

    def __init__(self, in_dim, dist_dim, in_length=1):
        """ Models the axis-aligned Gaussian distribution
            arguments:
            :param in_dim: number of filters in the input tensor
            :param dist_dim: number of dimension in the output distribution
            :param in_length: length of input tensor list
        """
        super().__init__()

        n_filters = 52  # arXiv:2003.06409v2
        kernel_size = 3

        self.in_length = in_length
        self.dist_dim = dist_dim

        self.conv1 = nn.Conv2d(in_dim, n_filters, kernel_size, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(n_filters)
        self.conv2 = nn.Conv2d(n_filters, n_filters, kernel_size, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(n_filters)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_dim, n_filters, kernel_size=1, stride=4, bias=False),
            nn.BatchNorm2d(n_filters)
        )

        self.linear = nn.Linear(n_filters*in_length, dist_dim*2)

    def forward(self, x):
        """ forward pass of DiagonalGaussian
        args:
            x: dynamics feature, i.e. tensor of shape [b x C x H x W] or list of tensor of such shape (in case of future distrubition)
        output:
            dist: a normal distribution
        """
        assert isinstance(x, list) or self.in_length == 1

        if self.in_length == 1:
            input = [x]
        else:
            assert len(x) == self.in_length, "Input length should be {}, got {}".format(self.in_length, len(x))
            input = x

        out_features = []

        for x in input:
            identity = x
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.bn2(x)
            identity = self.downsample(identity)
            x += identity
            x = self.relu(x)

            in_shape = x.shape[-2:]
            x = F.avg_pool2d(x, in_shape)
            x = torch.flatten(x, start_dim=1)
            out_features.append(x)

        x = torch.cat(out_features, dim=1)

        linear_out = self.linear(x)
        mu, log_sigma = torch.split(linear_out, self.dist_dim, dim=1)
        sigma = torch.exp(log_sigma)

        dist = Independent(Normal(loc=mu, scale=sigma), reinterpreted_batch_ndims=1)

        return dist


class ConditioningDistributions(nn.Module):

    def __init__(self, latent_dim, dist_dim, n_latent_past=1, n_latent_future=1):
        """
        args:
            latent_dim: dimension C as in the latent tensor z [b x C x H x W]
            dist_dim: dimension L as in mu and sigma: [b x L]
            n_latent_past: number of latent tensors to encode the past frames
            n_latent_future: number of latent tensors to encode the past and future frames, this is needed only as a teacher during training
        """
        super().__init__()

        self.present_dist = DiagonalGaussian(latent_dim, dist_dim, n_latent_past)
        self.future_dist = DiagonalGaussian(latent_dim, dist_dim, n_latent_future)

    def forward(self, z_present, z_future=None):
        out_present = self.present_dist(z_present)

        if self.training:
            assert z_future is not None, "Future latent encoding needed during training!"
            out_future = self.future_dist(z_future)
        else:
            out_future = None

        return {'present_dist': out_present, 'future_dist': out_future}
