import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class ConvGRUCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, activation=torch.sigmoid):
        """
        Arguments:
            input_dim: number of channels of input tensor
            hidden_dim: number of channels of hidden state
            kernel_size: convolutional kernel size
            activation: unlinearities for reset / update / out gates
        """
        super().__init__()
        padding = kernel_size // 2
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.reset_gate = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=padding)
        self.update_gate = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=padding)
        self.out_gate = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=padding)
        self.activation = activation

        init.orthogonal_(self.reset_gate.weight)
        init.orthogonal_(self.update_gate.weight)
        init.orthogonal_(self.out_gate.weight)
        init.constant_(self.reset_gate.bias, 0.)
        init.constant_(self.update_gate.bias, 0.)
        init.constant_(self.out_gate.bias, 0.)

    def forward(self, x, h):
        """
        Forward pass of ConvGRUCell
        :param x: dictionary of tensors. Keys: 'input' and 'hidden', tensor of shape b x C x H x W. Should be of same spatial size.
        """
        assert x.shape[-2:] == h.shape[-2:], "x and h should have same spatial shape."

        stacked_inputs = torch.cat([x, h], dim=1)

        update = self.activation(self.update_gate(stacked_inputs))
        reset = self.activation(self.reset_gate(stacked_inputs))
        out_inputs = torch.relu(self.out_gate(torch.cat([x, h * reset], dim=1)))
        new_state = h * (1 - update) + out_inputs * update

        return new_state


class ConvGRU(nn.Module):
    """ Unwraps input in the time dimension and pass it into ConvGRUCell """

    def __init__(self, input_dim, hidden_dim, kernel_size):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cell = ConvGRUCell(input_dim, hidden_dim, kernel_size)

    def forward(self, x, h=None):
        """
        :param x: dictionary of tensor, keys:
            'input': input tensor of shape [b x input_dim x T x H x W]
            'hidden': initial hidden state of shape [b x hidden_dim x H x W], or None
        return:
            'hidden': hidden state after each timestamp of shape [b x hidden_dim x T x H x W]
        """
        assert len(x.shape) == 5, "Input should have shape b x T x C x H x W, got {}".format(x.shape)
        if h is None:
            h = torch.zeros(x.shape[0], self.hidden_dim, *x.shape[-2:]).to(x) # b x hidden_dim x H x W

        h_out = []
        for t in range(x.shape[2]):
            h = self.cell(x[:, :, t, :, :], h)
            h_out.append(h)

        h_out = torch.stack(h_out, dim=2)
        return h_out


class Dynamics(nn.Module):

    def __init__(self, in_channels, k_spatial, n_frames, layer_channels):
        """ Wrapper Class for Temporal Block Dynamics Module

            args:
                in_channels: number of input features
                k_spatial: size of the spatial kernel
                n_frames: number of frames to be processed by the dynamics module
                layer_channels: list of output channels per layer, with the length of list equal to n_frames-1

            module input shape:
                batch x channel x T x H x W

            module output shape:
                batch x Cout x H x W
        """
        super().__init__()
        assert n_frames > 1, "n_frames should > 1, got {}".format(n_frames)

        conv_layers = []
        last_ch = in_channels
        for layer_ch in layer_channels:
            conv_layers.append(
                ConvGRU(last_ch, layer_ch, 3)
            )
            last_ch = layer_ch
        self.conv_layers = nn.Sequential(*conv_layers)

    def forward(self, x):
        T = x.shape[2]

        x = self.conv_layers(x)

        return x[:, :, -1, :, :]


if __name__ == "__main__":
    dyn = Dynamics(10, 3, 5, [80, 88, 96, 104])
    a = torch.zeros((2, 10, 5, 33, 65))
    b = dyn(a)
    print(b.shape)
