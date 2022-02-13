import torch
import torch.nn as nn
import torch.nn.functional as F

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__()
        self.aspp_pooling = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        size = x.shape[-2:]
        x = self.aspp_pooling(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=True)


class ASPP(nn.Module):
    """ Atrous Spatial Pyramid Pooling (ASPP) module as proposed in arXiv:1706.05587v3 """

    def __init__(self, in_channels, conv_channels, atrous_rates, out_channels=None):
        """ Arguments:
            in_channels: Integer, input channels
            conv_channels: Integer, output channels of each convolution. Please note that the final dimension = (2+len(astrous_rates)) * out_channels if ``project_channels`` is not given.
            astrous_rates: tuple of astrous rates. The final output will concatenate the output from one 1x1 convolution, one pooling layer and all astrous 3x3 covolutions.
            out_channels: if given, project the concatenated feature into ``out_channels`` dimension
        """
        super(ASPP, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, conv_channels, 1, bias=False),
            nn.BatchNorm2d(conv_channels),
            nn.ReLU(inplace=True)))

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, conv_channels, rate))

        modules.append(ASPPPooling(in_channels, conv_channels))

        self.convs = nn.ModuleList(modules)

        if out_channels:
            n_module = len(modules)
            self.project = nn.Sequential(
                nn.Conv2d(n_module * conv_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
                # nn.Dropout(0.1)
            )
        else:
            self.project = nn.Identity()

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        res = self.project(res)
        return res
