import torch
import torch.nn as nn
import torch.nn.functional as F

class FutureDecoderTransConv(nn.Module):

    def __init__(self, in_features, out_features, key_feature,
                 use_skip_con=False, skip_features=[], skip_features_proj=[], skip_keys=[],
                 odd_upsampling=False):
        """ Arguments:
            :param in_features: input features
            :param out_features: output features of the task head. e.g. should be the number of classes for segmentation task
            :param key_feature: string, key for high level feature
            :param use_skip_con: if use skip connection from encoder. Default: False
            :param use_skip_con: wether to use skip (residual) connection from encoder
            :param skip_features: number of features for each skip connection (with high-level feature first)
            :param skip_features_proj: number of features to project for each skip connection (with high-level feature first)
            :param skip_keys: list of keys for each skip connection (with high-level feature first)
            :param odd_upsampling: if True, will upsample shape k into 2*k-1
        """

        super().__init__()

        self.use_skip_con = use_skip_con
        self.key_feature = key_feature
        self.skip_keys = skip_keys
        self.odd_upsampling = odd_upsampling

        if use_skip_con:
            assert len(skip_features) == len(skip_features_proj) == len(skip_keys) == 2, f"Only 2 skip connections are supported in this decoder!"
            SKIP_PLANES1_PROJ = skip_features_proj[0]
            SKIP_PLANES2_PROJ = skip_features_proj[1]
            self.convp1 = nn.Conv2d(skip_features[0], SKIP_PLANES1_PROJ, 1, bias=False)
            self.bnp1 = nn.BatchNorm2d(SKIP_PLANES1_PROJ)
            self.convp2 = nn.Conv2d(skip_features[1], SKIP_PLANES2_PROJ, 1, bias=False)
            self.bnp2 = nn.BatchNorm2d(SKIP_PLANES2_PROJ)
        else:
            SKIP_PLANES1_PROJ = 0
            SKIP_PLANES2_PROJ = 0

        INPLANES1 = 64
        INPLANES2 = 32
        self.upconv1 = nn.ConvTranspose2d(in_features, INPLANES1, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(INPLANES1)
        self.upconv2 = nn.ConvTranspose2d(INPLANES1+SKIP_PLANES1_PROJ, INPLANES1, 3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(INPLANES1)
        self.conv3 = nn.Conv2d(INPLANES1+SKIP_PLANES2_PROJ, INPLANES2, 3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(INPLANES2)

        self.relu = nn.ReLU(inplace=True)
        self.project = nn.Conv2d(INPLANES2, out_features, 1, bias=True)

    def forward(self, x):
        identity = x
        x = x[self.key_feature] # compatible with panoptic-deeplab

        upsample_size = lambda x: (x.shape[-2]*2-1, x.shape[-1]*2-1) if self.odd_upsampling else None

        x = self.upconv1(x, upsample_size(x))
        x = self.bn1(x)
        x = self.relu(x)

        if self.use_skip_con:
            skip = identity[self.skip_keys[0]]
            skip = self.convp1(skip)
            skip = self.bnp1(skip)
            skip = self.relu(skip)
            x = torch.cat([x, skip], dim=1)

        x = self.upconv2(x, upsample_size(x))
        x = self.bn2(x)
        x = self.relu(x)

        if self.use_skip_con:
            skip = identity[self.skip_keys[1]]
            skip = self.convp2(skip)
            skip = self.bnp2(skip)
            skip = self.relu(skip)
            x = torch.cat([x, skip], dim=1)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.project(x)

        return x


class FutureDecoderUpConv(nn.Module):
    """ Future decoder with upsampling + 1x1covolution instead of transposed convolution """

    def __init__(self, in_features, out_features, key_feature, odd_upsampling=False):
        """ Arguments:
            :param in_features: input features
            :param out_features: output features of the task head. e.g. should be the number of classes for segmentation task
            :param key_feature: string, key for high level feature
            :param odd_upsampling: bool, if True, a dimension k will be upsampled as 2k-1
        """

        super().__init__()

        self.key_feature = key_feature
        self.odd_upsampling = odd_upsampling

        INPLANES1 = 64
        INPLANES2 = 32
        self.conv1 = nn.Conv2d(in_features, INPLANES1, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(INPLANES1)
        self.conv2 = nn.Conv2d(INPLANES1, INPLANES1, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(INPLANES1)
        self.conv3 = nn.Conv2d(INPLANES1, INPLANES2, 3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(INPLANES2)

        self.relu = nn.ReLU(inplace=True)
        self.project = nn.Conv2d(INPLANES2, out_features, 1, bias=True)

    def forward(self, x):
        identity = x
        x = x[self.key_feature] # compatible with panoptic-deeplab

        if self.odd_upsampling:
            align_corners = True
            upsampled_size = lambda x: (x.shape[-2]*2-1, x.shape[-1]*2-1)
        else:
            align_corners = False
            upsampled_size = lambda x: (x.shape[-2]*2, x.shape[-1]*2)

        x = F.interpolate(x, size=upsampled_size(x), mode='bilinear', align_corners=align_corners)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = F.interpolate(x, size=upsampled_size(x), mode='bilinear', align_corners=align_corners)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.project(x)

        return x

class ControlDecoder(nn.Module):
    ### TODO

    def __init__(self):
        pass

    def forward(self, x):
        pass
