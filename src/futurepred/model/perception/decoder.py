import torch
import torch.nn as nn
import torch.nn.functional as F

from . import aspp

class SingleFrameDecoder(nn.Module):
    """ Decoder for single-frame semantic segmentaiton. This module is used for pretraining the perception encoder. """

    @staticmethod
    def tconv1x1(in_planes, out_planes, stride=1):
        return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                bias=False, padding=0)

    @staticmethod
    def tconv3x3(in_planes, out_planes, stride=1):
        return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=stride,
                                bias=False, padding=1)

    @staticmethod
    def tconv5x5(in_planes, out_planes, stride=1):
        return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=5, stride=stride,
                                bias=False, padding=2)

    class BasicBlock(nn.Module):
        expansion = 1

        def __init__(self, inplanes, planes, stride=1, norm_layer=None, even_upsample=True, use_rescon=False):
            super().__init__()
            if norm_layer is None:
                norm_layer = nn.BatchNorm2d
            self.tconv1 = SingleFrameDecoder.tconv5x5(inplanes, planes, stride)
            self.bn1 = norm_layer(planes)
            self.relu = nn.ReLU(inplace=True)
            self.tconv2 = SingleFrameDecoder.tconv5x5(planes, planes)
            self.bn2 = norm_layer(planes)

            if use_rescon and (stride!= 1 or inplanes != planes):
                self.tconv3 = SingleFrameDecoder.tconv1x1(inplanes, planes, stride)
                self.bn3 = norm_layer(planes)
                self.upsample = True
            else:
                self.upsample = False
            self.use_rescon = use_rescon

            self.stride = stride
            self.even_upsample = even_upsample

        def forward(self, x):

            if self.even_upsample:
                out_shape = tuple([l*self.stride for l in x.shape[-2:]])

            if self.use_rescon:
                identity = x

            out = self.tconv1(x, out_shape)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.tconv2(out)
            out = self.bn2(out)

            if self.use_rescon:
                if self.upsample:
                    identity = self.tconv3(x, out_shape)
                    identity = self.bn3(identity)

                out += identity

            out = self.relu(out)

            return out

    class SimpleTransConv2d(nn.Module):
        expansion = 1

        def __init__(self, inplanes, planes, stride=1, even_upsample=True):
            super().__init__()
            self.tconv1 = SingleFrameDecoder.tconv3x3(inplanes, planes, stride)
            self.bn = nn.BatchNorm2d(planes)
            self.relu = nn.ReLU(inplace=True)
            self.even_upsample = even_upsample
            self.stride = stride

        def forward(self, x):
            if self.even_upsample:
                out_shape = tuple([l*self.stride for l in x.shape[-2:]])
            else:
                out_shape = None

            x = self.tconv1(x, out_shape)
            x = self.bn(x)
            x = self.relu(x)

            return x

    def __init__(self, in_features, out_features, key_feature,
                 decoder_block=BasicBlock, n_blocks=[], decoder_features=[],
                 use_skip_con=False, skip_type='cat', skip_features=[], skip_features_proj=[], skip_keys=[],
                 out_activation=nn.Identity()):
        """ Arguments:
            in_features (int): number of input filters
            out_features (int): number of output classes
            key_feature (str): highest level feature key from the encoder
            decoder_block (nn.Module): type decoder block. Currently only ``BasicBlock`` is supported
            n_blocks (int): list specifying number of decoder blocks in each layer
            decoder_features (list): list containing the number of filters to use in each layer
            use_skip_con (Bool): if True, use skip connection from encoder
            skip_type (str): ``cat`` for concatenate, or ``sum`` for addition
            skip_features (list): number of filters in each skip connection with low resolution (high level feature) first
            skip_features_proj (list): number of filters after projecting each skip connection
            skip_keys (list): keys to use for each skip connection with low resolution first
            out_activation (nn.Module): activation before the final output
        """

        super().__init__()

        self._norm_layer = nn.BatchNorm2d

        self.inplanes = in_features
        self.key_feature = key_feature
        self.skip_keys = skip_keys
        self.use_skip_con = use_skip_con
        self.skip_type = skip_type

        self.layers = []
        self.project_skip = []

        assert len(n_blocks) > 1, "At least one decoder block is needed for this module!"
        assert len(n_blocks) == len(decoder_features)
        assert (not use_skip_con) or (len(skip_features) == len(skip_features_proj) == len(skip_keys) == len(n_blocks))

        for idx, (n_block, n_channel) in enumerate(zip(n_blocks, decoder_features)):
            self.layers.append(self._make_layer_decoder(decoder_block, n_channel, n_block, stride=2))
            if use_skip_con:
                if skip_type == 'cat':
                    if skip_features[idx] and skip_features_proj[idx] and skip_keys[idx]:
                        self.project_skip.append(nn.Conv2d(skip_features[idx], skip_features_proj[idx], 1))
                        self.inplanes = n_channel + skip_features_proj[idx]
                    else:
                        # no skip connection made in level idx
                        self.project_skip.append(None)
                        self.inplanes = n_channel
                # elif skip_type == 'sum':
                #     if skip_features[idx] and skip_features_proj[idx] and skip_keys[idx]:
                #         assert skip_features_proj[idx] == n_channel, "For additive skip connection, projected channel must equal decoder features in each level!"
                #         if skip_features[idx] == skip_features_proj[idx]:
                #             project = nn.Identity()
                #         else:
                #             project = nn.Conv2d(skip_features[idx], skip_features_proj[idx], 1)
                #         self.project_skip.append(project)
                #         self.inplanes = n_channel
                #     else:
                #         # no skip connection made in level idx
                #         self.project_skip.append(None)
                #         self.inplanes = n_channel
                else:
                    raise ValueError(f"Unsupported skip_type={skip_type}!")

        self.layers = nn.ModuleList(self.layers)
        if use_skip_con:
            self.project_skip = nn.ModuleList(self.project_skip)

        # ASPP_CHANNELS = 256
        # self.aspp = aspp.ASPP(self.inplanes, conv_channels=ASPP_CHANNELS, atrous_rates=(12,24,36), out_channels=256)

        OUT_CONV_CHANNELS = 256
        self.out_conv = nn.Sequential(
            nn.Conv2d(self.inplanes, OUT_CONV_CHANNELS, kernel_size=3, padding=2, bias=False),
            nn.BatchNorm2d(OUT_CONV_CHANNELS),
            nn.ReLU(inplace=True),
            nn.Conv2d(OUT_CONV_CHANNELS, out_features, kernel_size=1, bias=True)
        )

        self.out_activation = out_activation

    def forward(self, x):
        """
        args:
            x: encoded representation of input image, shape [b x Ch x H x W]
            res_con: a list with length 3. The skip connection from the last 3 blocks
        output:
            out: shape [b x out_features x H x W]. The H and W are upsampled 8 times here
        """
        out = x[self.key_feature]

        for idx, layer in enumerate(self.layers):
            out = layer(out)
            if self.use_skip_con and self.project_skip[idx]:
                skip = self.project_skip[idx](x[self.skip_keys[idx]])
                if self.skip_type == 'cat':
                    out = torch.cat([out, skip], dim=1) # TODO: not sure if it's summing or concatenate. Need to try both
                elif self.skip_type == 'sum':
                    out += skip
                else:
                    raise ValueError(f"Unsupported skip type!")

        # out = self.aspp(out)
        out = self.out_conv(out)
        out = self.out_activation(out)

        return out


    def _make_layer_decoder(self, block, planes, blocks, stride=1):
        """Make blocks number of block
        planes: number of output features
        """
        norm_layer = self._norm_layer

        layers = []
        layers.append(block(self.inplanes, planes, stride))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, stride=1))

        return nn.Sequential(*layers)
