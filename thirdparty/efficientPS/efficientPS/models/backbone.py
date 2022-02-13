import sys
from collections import OrderedDict
from functools import partial

import torch.nn as nn
from inplace_abn import ABN
from torch.nn import functional as F
import torch
from efficientPS.modules.misc import GlobalAvgPool2d
from efficientPS.modules.residual import ResidualBlock
from efficientPS.utils.misc import try_index

from .utils import (
    round_filters,
    round_repeats,
    drop_connect,
    get_same_padding_conv2d,
    get_model_params,
    efficientnet_params,
    load_pretrained_weights,
    Swish,
    MemoryEfficientSwish,
)


class ResNet(nn.Module):
    """Standard residual network

    Parameters
    ----------
    structure : list of int
        Number of residual blocks in each of the four modules of the network
    bottleneck : bool
        If `True` use "bottleneck" residual blocks with 3 convolutions, otherwise use standard blocks
    norm_act : callable or list of callable
        Function to create normalization / activation Module. If a list is passed it should have four elements, one for
        each module of the network
    classes : int
        If not `0` also include global average pooling and a fully-connected layer with `classes` outputs at the end
        of the network
    dilation : int or list of int
        List of dilation factors for the four modules of the network, or `1` to ignore dilation
    dropout : list of float or None
        If present, specifies the amount of dropout to apply in the blocks of each of the four modules of the network
    caffe_mode : bool
        If `True`, use bias in the first convolution for compatibility with the Caffe pretrained models
    """

    def __init__(self,
                 structure,
                 bottleneck,
                 norm_act=ABN,
                 classes=0,
                 dilation=1,
                 dropout=None,
                 caffe_mode=False):
        super(ResNet, self).__init__()
        self.structure = structure
        self.bottleneck = bottleneck
        self.dilation = dilation
        self.dropout = dropout
        self.caffe_mode = caffe_mode

        if len(structure) != 4:
            raise ValueError("Expected a structure with four values")
        if dilation != 1 and len(dilation) != 4:
            raise ValueError("If dilation is not 1 it must contain four values")

        # Initial layers
        layers = [
            ("conv1", nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=caffe_mode)),
            ("bn1", try_index(norm_act, 0)(64))
        ]
        if try_index(dilation, 0) == 1:
            layers.append(("pool1", nn.MaxPool2d(3, stride=2, padding=1)))
        self.mod1 = nn.Sequential(OrderedDict(layers))

        # Groups of residual blocks
        in_channels = 64
        if self.bottleneck:
            channels = (64, 64, 256)
        else:
            channels = (64, 64)
        for mod_id, num in enumerate(structure):
            mod_dropout = None
            if self.dropout is not None:
                if self.dropout[mod_id] is not None:
                    mod_dropout = partial(nn.Dropout, p=self.dropout[mod_id])

            # Create blocks for module
            blocks = []
            for block_id in range(num):
                stride, dil = self._stride_dilation(dilation, mod_id, block_id)
                blocks.append((
                    "block%d" % (block_id + 1),
                    ResidualBlock(in_channels, channels, norm_act=try_index(norm_act, mod_id),
                                  stride=stride, dilation=dil, dropout=mod_dropout)
                ))

                # Update channels and p_keep
                in_channels = channels[-1]

            # Create module
            self.add_module("mod%d" % (mod_id + 2), nn.Sequential(OrderedDict(blocks)))

            # Double the number of channels for the next module
            channels = [c * 2 for c in channels]

        # Pooling and predictor
        if classes != 0:
            self.classifier = nn.Sequential(OrderedDict([
                ("avg_pool", GlobalAvgPool2d()),
                ("fc", nn.Linear(in_channels, classes))
            ]))

    @staticmethod
    def _stride_dilation(dilation, mod_id, block_id):
        d = try_index(dilation, mod_id)
        s = 2 if d == 1 and block_id == 0 and mod_id > 0 else 1
        return s, d

    def forward(self, x):
        outs = OrderedDict()

        outs["mod1"] = self.mod1(x)
        outs["mod2"] = self.mod2(outs["mod1"])
        outs["mod3"] = self.mod3(outs["mod2"])
        outs["mod4"] = self.mod4(outs["mod3"])
        outs["mod5"] = self.mod5(outs["mod4"])

        if hasattr(self, "classifier"):
            outs["classifier"] = self.classifier(outs["mod5"])

        return outs

class MBConvBlock(nn.Module):
    """
    Mobile Inverted Residual Bottleneck Block
    Args:
        block_args (namedtuple): BlockArgs, see above
        global_params (namedtuple): GlobalParam, see above
    Attributes:
        has_se (bool): Whether the block contains a Squeeze and Excitation layer.
    """

    def __init__(self, block_args, global_params, norm_act=ABN, dilation=1):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip  # skip connection and drop connect

        # Get static or dynamic convolution depending on image size
        Conv2d = get_same_padding_conv2d(image_size=None)

        # Expansion phase
        inp = self._block_args.input_filters  # number of input channels
        oup = self._block_args.input_filters * self._block_args.expand_ratio  # number of output channels
        if self._block_args.expand_ratio != 1:
            self._expand_conv = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = norm_act(oup)
            self._bn0.activation = "identity"
        # Depthwise convolution phase
        k = self._block_args.kernel_size
        s = self._block_args.stride #[self._block_args.stride, self._block_args.stride]
        self._depthwise_conv = Conv2d(
            in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
            kernel_size=k, stride=s, bias=False, dilation=dilation)
        self._bn1 = norm_act(oup)
        self._bn1.activation = "identity"
        # Squeeze and Excitation layer, if desired
#        if self.has_se:
#            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
#            self._se_reduce = Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
#            self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        # Output phase
        final_oup = self._block_args.output_filters
        self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = norm_act(final_oup)
        self._bn2.activation ="identity"
        self._swish = MemoryEfficientSwish()

    def forward(self, inputs, drop_connect_rate=None):
        """
        :param inputs: input tensor
        :param drop_connect_rate: drop connect rate (float, between 0 and 1)
        :return: output of block
        """

        # Expansion and Depthwise Convolution
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._swish(self._bn0(self._expand_conv(inputs))) #self._swish
        x = self._swish(self._bn1(self._depthwise_conv(x))) #self._swish

        # Squeeze and Excitation
 #       if self.has_se:
 #           x_squeezed = F.adaptive_avg_pool2d(x, 1)
 #           x_squeezed = self._se_expand(self._swish(self._se_reduce(x_squeezed)))
 #           x = torch.sigmoid(x_squeezed) * x

        x = self._bn2(self._project_conv(x))

        # Skip connection and drop connect
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export)"""
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()


class EfficientNet(nn.Module):
    """
    An EfficientNet model. Most easily loaded with the .from_name or .from_pretrained methods
    Args:
        blocks_args (list): A list of BlockArgs to construct blocks
        global_params (namedtuple): A set of GlobalParams shared between blocks
    Example:
        model = EfficientNet.from_pretrained('efficientnet-b0')
    """

    def __init__(self, model_name='efficientnet-b7',norm_act=ABN, dilation=[1,1,1,1,1,1,1]):
        super().__init__()
        blocks_args, global_params = get_model_params(model_name, None)
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self._global_params = global_params
        self._blocks_args = blocks_args
        self.mod_idx = []
        self.corresponding_channels=[]

        # Get static or dynamic convolution depending on image size
        Conv2d = get_same_padding_conv2d(image_size=None)

        # Stem
        in_channels = 3  # rgb
        out_channels = round_filters(32, self._global_params)  # number of output channels

        self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=[2,2], bias=False)
        self._bn0 = norm_act(out_channels)
        self._bn0.activation = "identity"

        # Build blocks
        self._blocks = nn.ModuleList([])
        for block_args, dil in zip(self._blocks_args,dilation):
            self.corresponding_channels.append(round_filters(block_args.input_filters, self._global_params))
            if dil==2:
                block_args = block_args._replace(stride=1)

            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self._global_params),
                output_filters=round_filters(block_args.output_filters, self._global_params),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params)
            )

            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(MBConvBlock(block_args, self._global_params, norm_act, dil))
            if block_args.num_repeat > 1:
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(block_args, self._global_params, norm_act, dil))
            self.mod_idx.append(len(self._blocks)-1)

        # Head
        in_channels = block_args.output_filters  # output of final block
        out_channels = round_filters(1280, self._global_params)
        self.corresponding_channels.append(out_channels)
        self._conv_head = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1 =norm_act(out_channels)
        self._bn1.activation = "identity"

        self._swish = MemoryEfficientSwish()

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export)"""
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()
        for block in self._blocks:
            block.set_swish(memory_efficient)


    def extract_features(self, inputs):
        """ Returns output of the final convolution layer """

        # Stem
        outs = OrderedDict()
        name_mod = ['mod2', 'mod3','mod4','mod5']
        curr = 0
        x = self._swish(self._bn0(self._conv_stem(inputs))) #self._swish

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            if idx in self.mod_idx[1:3] or idx == self.mod_idx[4]:
                outs[name_mod[curr]]=x
                curr+=1
        # Head

        x = self._swish(self._bn1(self._conv_head(x))) #self._swish
        outs[name_mod[curr]]=x
        return outs

    def forward(self, inputs):
        """ Calls extract_features to extract features, applies final linear layer, and returns logits. """
        bs = inputs.size(0)
        # Convolution layers
        x = self.extract_features(inputs)
        return x

    @classmethod
    def from_name(cls, model_name, override_params=None):
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = get_model_params(model_name, override_params)
        return cls(blocks_args, global_params)

    @classmethod
    def from_pretrained(cls, model_name, num_classes=1000, in_channels = 3):
        model = cls.from_name(model_name, override_params={'num_classes': num_classes})
        load_pretrained_weights(model, model_name, load_fc=(num_classes == 1000))
        if in_channels != 3:
            Conv2d = get_same_padding_conv2d(image_size = model._global_params.image_size)
            out_channels = round_filters(32, model._global_params)
            model._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        return model

    @classmethod
    def from_pretrained(cls, model_name):
        print (cls, model_name)
        model = cls.from_name(model_name)
        load_pretrained_weights(model, model_name)

        return model

    @classmethod
    def get_image_size(cls, model_name):
        cls._check_model_name_is_valid(model_name)
        _, _, res, _ = efficientnet_params(model_name)
        return res

    @classmethod
    def _check_model_name_is_valid(cls, model_name, also_need_pretrained_weights=False):
        """ Validates model name. None that pretrained weights are only available for
        the first four models (efficientnet-b{i} for i in 0,1,2,3) at the moment. """
        num_models = 4 if also_need_pretrained_weights else 8
        valid_models = ['efficientnet-b'+str(i) for i in range(num_models)]
        if model_name not in valid_models:
            raise ValueError('model_name should be one of: ' + ', '.join(valid_models))

_NETS = {
    "18": {"structure": [2, 2, 2, 2], "bottleneck": False},
    "34": {"structure": [3, 4, 6, 3], "bottleneck": False},
    "50": {"structure": [3, 4, 6, 3], "bottleneck": True},
    "101": {"structure": [3, 4, 23, 3], "bottleneck": True},
    "152": {"structure": [3, 8, 36, 3], "bottleneck": True},
}
_ENETS = {
    "efficientnet-b0": {"model_name": "efficientnet-b0"},
    "efficientnet-b1": {"model_name": "efficientnet-b1"},
    "efficientnet-b2": {"model_name": "efficientnet-b2"},
    "efficientnet-b3": {"model_name": "efficientnet-b3"},
    "efficientnet-b4": {"model_name": "efficientnet-b4"},
    "efficientnet-b5": {"model_name": "efficientnet-b5"},
    "efficientnet-b6": {"model_name": "efficientnet-b6"},
    "efficientnet-b7": {"model_name": "efficientnet-b7"},
}

__all__ = []
for name, params in _NETS.items():
    net_name = "net_resnet" + name
    setattr(sys.modules[__name__], net_name, partial(ResNet, **params))
    __all__.append(net_name)

for name, params in _ENETS.items():
    net_name = "net_" + name
    setattr(sys.modules[__name__], net_name, partial(EfficientNet, **params))
    __all__.append(net_name)

#if __name__=='__main__':
#   blocks_args, global_params = get_model_params('efficientnet-b7', None)
#   a = EfficientNet(blocks_args, global_params)
