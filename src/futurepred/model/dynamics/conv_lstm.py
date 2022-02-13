### This implementation follows the LSTM formulation from Nabavi et al (arXiv: arXiv:1807.07946v1)

import torch.nn as nn
import torch
from torch.nn.parameter import Parameter


class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, input_shape):
        """
        Args:
            input_dim (int): number of channels for input tensor
            hidden_dim (int): number of channels for hidden state
            kernel_size (int): kernel size of conv2d layer
            input_shape (tuple[int, int]): shape of the last two dimensions of
                input tensor
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2

        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=True
        )

        self.weight_cell_input = Parameter(torch.zeros(hidden_dim, *input_shape))
        self.weight_cell_forget = Parameter(torch.zeros(hidden_dim, *input_shape))
        self.weight_cell_output = Parameter(torch.zeros(hidden_dim, *input_shape))


    def forward(self, input_tensor, cur_state):
        """
        Args:
            input_tensor (Tensor[b, Cin, H, W]): input tensor
            cur_state (list(Tensor[b, Chid, H, W], Tensor[b, Chid, H, W])): last cell output and hidden state

        Returns:
            list(Tensor[b, Chid, H, W], Tensor[b, Chid, H, W]): next hidden state and next cell output
        """
        hidden_cur, cell_output_cur = cur_state

        combined = torch.cat([input_tensor, hidden_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        input_gate, forget_gate, cell_output, output_gate = torch.split(
            combined_conv,
            self.hidden_dim,
            dim=1
        )

        input_gate = torch.sigmoid(self.weight_cell_input * cell_output_cur + input_gate)
        forget_gate = torch.sigmoid(self.weight_cell_forget * cell_output_cur + forget_gate)
        cell_output = forget_gate * cell_output_cur + input_gate * torch.tanh(cell_output)
        output_gate = torch.sigmoid(output_gate + self.weight_cell_output * cell_output)
        hidden = output_gate * torch.tanh(cell_output)

        return hidden, cell_output

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (
            torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
            torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device)
        )


class ConvLSTM(nn.Module):

    """
    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.
    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, input_shape, num_layers):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.input_shape = input_shape
        self.num_layers = num_layers

        cell_list = []
        for i in range(self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(
                ConvLSTMCell(
                    input_dim=cur_input_dim,
                    hidden_dim=self.hidden_dim[i],
                    kernel_size=self.kernel_size[i],
                    input_shape=self.input_shape,
                )
            )

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (B, C, T, H, W)
        hidden_state: todo
            None. todo implement stateful
        -------
        Returns
            list(Tensor[B, C_hid, H, W], Tensor[B, C_hid, H, W]): hidden state and cell output of last layer
        """
        b, _, _, h, w = input_tensor.size()

        assert (h, w) == tuple(self.input_shape), f"Invalid input shape {(h, w)}"

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=self.input_shape)

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(2)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, :, t, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=2)
            cur_layer_input = layer_output

            # layer_output_list.append(layer_output)
            # last_state_list.append([h, c])

        # if not self.return_all_layers:
        #     layer_output_list = layer_output_list[-1:]
        #     last_state_list = last_state_list[-1:]

        # return layer_output_list, last_state_list
        return layer_output

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class Dynamics(ConvLSTM):

    def __init__(self, in_channels, in_shape, k_spatial, layer_channels):

        n_layers = len(layer_channels)

        super().__init__(
            in_channels,
            layer_channels,
            (k_spatial, k_spatial),
            in_shape,
            n_layers
        )

    def forward(self, x):
        x = super().forward(x)
        return x[:, :, -1, :, :]


if __name__ == '__main__':

    convlstm = ConvLSTM(10, 20, (3, 3), (33, 65), 4)
    in_tensor = torch.zeros(3, 10, 5, 33, 65)
    out_tensor = convlstm(in_tensor)

    print(out_tensor.shape)

    dynamics = Dynamics(10, [33, 65], 3, [80, 88, 96, 104])
    out_tensor = dynamics(in_tensor)

    print(out_tensor.shape)
