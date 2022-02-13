import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from .conv_gru import ConvGRU
from .normalization import ConditionalBatchNorm2d

class ResidualConv2dBlock(nn.Module):

    def __init__(self, in_features, out_features, dim_cond, n_frame):
        super().__init__()

        norm_layer = ConditionalBatchNorm2d

        self.bn1 = norm_layer(in_features, dim_cond, n_frame)
        self.conv1 = self.conv3x3(in_features, out_features)
        self.bn2 = norm_layer(out_features, dim_cond, n_frame)
        self.conv2 = self.conv3x3(out_features, out_features)
        self.bn3 = norm_layer(out_features, dim_cond, n_frame)
        self.conv3 = self.conv3x3(out_features, out_features)
        self.relu = nn.ReLU(inplace=True)

        self.conv_ds = self.conv1x1(in_features, out_features)

    def forward(self, x, cond):
        # x: feature, c: conditioning signal (noise vector)
        identity = x

        out = self.bn1(x, cond)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out, cond)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out, cond)
        out = self.relu(out)
        out = self.conv3(out)

        identity = self.conv_ds(identity)

        out += identity
        out = self.relu(out)

        return out

    @staticmethod
    def conv3x3(in_planes, out_planes, stride=1):
        return spectral_norm(nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False))

    @staticmethod
    def conv1x1(in_planes, out_planes):
        return spectral_norm(nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False))


class _GeneratorLayer(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, cond_dim, n_frame):
        """
        Defines one layer in the generator network.
        :param in_dim: dimension of input tensor
        :param hidden_dim: dimension hidden state, which equals to output dimension
        :param out_dim: output_dimension
        """
        super().__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.cond_dim = cond_dim

        self.gru = ConvGRU(in_dim, hidden_dim, kernel_size=3)
        self.conv_blk = ResidualConv2dBlock(hidden_dim, out_dim, cond_dim, n_frame)

    def forward(self, x):
        """
        :param x: dictionary of tensors -
            'input' : shape b x T x in_dim x H x W
            'hidden': shape b x hidden_dim x H x W or None. With None, an zero-tensor will be used as inital hidden state.
            'cond': b x cond_dim
        """
        b, T, in_dim, H, W = x['input'].shape
        cond = x['cond']

        out = self.gru(x)['hidden']

        out = out.view(-1, self.hidden_dim, H, W) # combine time and batch dimension - (bxT) x hidden_dim x H x W
        out = self.conv_blk(out, cond)
        out = out.view(b, T, self.out_dim, H, W)

        return {'input': out, 'hidden': None, 'cond': cond} # to be used by next layer


class Generator(nn.Module):

    def __init__(self, dynamics_dim, noise_dim, layer_dim_base, layer_dims, n_future):
        """ Arguments:
            dynamics_dim: Number of dimensions for the dynamic features - hidden state for the recurrent network
            noise_dim: Number of dimensions for the noise tensor (output from predictor module) - input for the recurrent network
            n_future: Number of future frames to predict
            dim_base: base number of layer dimensions
            layer_dims: list of dims in each layer, the final number of filters per layer will be layer_dims*dim_base
                e.g.:
                    dim_base = 96 # use 128 for dim 64x64 and 96 otherwise (https://arxiv.org/abs/1907.06571)
                    layer_dims = [8, 4, 2] # original author used 8, 8, 8, 4, 2 for 4x4-sized feature, which gives us output size of 128.
                                        # and since our dynamics feature is H/8 x W/8, we use three layers here
        """
        super().__init__()
        assert dynamics_dim == layer_dims[-1]*layer_dim_base, 'the dimension of generator output should equal to dynamics dimension!'

        self.n_future = n_future

        layers = []
        last_out_dim = noise_dim
        self.noise_dim = noise_dim
        for dim in layer_dims:
            out_dim = dim * layer_dim_base
            layer = _GeneratorLayer(last_out_dim, dynamics_dim, out_dim, noise_dim, n_future)
            layers.append(layer)
            last_out_dim = out_dim

        self.mod = nn.Sequential(*layers)

    def forward(self, x):
        """
        Arguments:
            x (dict):
                'dynamics': dynamics feature, with shape [b x dynamics_dim x H x W]
                'noise': noise tensor, with shape [b x noise_dim] or None. For deterministic prediction, feed None

        Output:
            z: generated dynamics feature representing timestep t+1, with shape [b x Cout x H x W]
        """
        dyn = x['dynamics']

        if x['noise'] is None:
            x['noise'] = torch.zeros(dyn.shape[0], self.noise_dim).to(dyn.device)
        noise = x['noise']
        cond = x['noise']

        # broadcast u across all HxW pixels, and repeat it n_future times
        noise = noise.unsqueeze(-1).unsqueeze(-1)
        size = dyn.shape[-2:]
        noise = F.interpolate(noise, size, mode='bilinear', align_corners=False).unsqueeze(1) # b x 1 x noise_dim x H x W
        noise = noise.repeat(1, self.n_future, 1, 1, 1) # b x T x noise_dim x H x W

        out = self.mod({'input': noise, 'hidden': dyn, 'cond': cond})

        return {'dynamics': out['input']}
