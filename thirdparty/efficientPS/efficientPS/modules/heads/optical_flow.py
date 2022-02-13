import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import ops
# import kornia
# from efficientPS.utils.optical_flow_ops import *
# import efficientPS.utils.optical_flow_correlation as correlation
# from efficientPS.modules.heads.correlation_package.correlation import Correlation
# from spatial_correlation_sampler import SpatialCorrelationSampler
# from efficientPS.modules.heads.bpnp import BPnP
# from efficientPS.modules.heads.dsacstar import DSACStar


class PreActBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, out_planes, stride=1, normalise=True):
        super(PreActBlock, self).__init__()
        if normalise:
            self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_planes != self.expansion * out_planes:
            conv_short = nn.Conv2d(in_planes, self.expansion * out_planes, kernel_size=1, stride=stride, bias=False)
            self.shortcut = nn.Sequential(conv_short)

    def forward(self, x):
        out = self.relu(self.bn1(x)) if hasattr(self, 'bn1') else x
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.relu(self.bn2(self.conv1(out)))
        out = self.conv2(out)

        out += shortcut
        return out


class HDADecoder(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(HDADecoder, self).__init__()
        self.block1 = PreActBlock(in_planes, out_planes, normalise=False)
        self.block2 = PreActBlock(out_planes, out_planes, normalise=True)

        self.root = nn.Sequential(nn.BatchNorm2d(out_planes * 2),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(out_planes * 2, out_planes, kernel_size=1, stride=1, bias=False))

    def forward(self, x):
        y1 = self.block1(x)
        y2 = self.block2(y1)
        out = self.root(torch.cat([y1, y2], 1))
        return out


class Decoder(nn.Module):

    def __init__(self, inplane, block, classes, up_classes):
        super(Decoder, self).__init__()
        self.mapping = block(inplane, 128)
        self.cls = nn.Sequential(
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(
                128, classes, kernel_size=1, stride=1, padding=0, bias=True))
        self.up = None
        if up_classes > 0:
            self.up = nn.Sequential(
                nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                nn.ConvTranspose2d(
                    128,
                    up_classes,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False), nn.BatchNorm2d(up_classes), nn.ReLU(inplace=True))

    def forward(self, x):
        out = self.mapping(x)
        prob = self.cls(out)
        up_feat = self.up(out) if self.up else None
        return prob, up_feat


class ChannelMapperBlock(nn.Module):
    """Convolve input channels to match output channels"""
    def __init__(self, in_channels, out_channels):
        super(ChannelMapperBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.conv1x1(x))
        return out


class OpticalFlowMapper(nn.Module):
    """Map the output of EfficientPS encoder to that of the flow decoder"""
    def __init__(self, in_channels_list, out_channels_list):
        super(OpticalFlowMapper, self).__init__()
        self.out_layers = len(out_channels_list)

        self.mapper_layer_list = nn.ModuleList()
        for in_ch, out_ch in zip(in_channels_list, out_channels_list):
            mapper_layer = ChannelMapperBlock(in_ch, out_ch)
            self.mapper_layer_list.append(mapper_layer)

    def forward(self, ms_feat):
        out_ms_feat = []
        ms_feat_idx = 0
        for l in range(self.out_layers):
            if l == 0:
                # Upsample the first two layers, concat them and then map their channels
                layer1, layer2 = ms_feat[0], ms_feat[1]
                layer1 = F.interpolate(layer1, scale_factor=2, mode="bilinear", align_corners=True, recompute_scale_factor=True)
                layer2 = F.interpolate(layer2, scale_factor=4, mode="bilinear", align_corners=True, recompute_scale_factor=True)
                mapped_layer = self.mapper_layer_list[l](torch.cat((layer1, layer2), dim=1))
                out_ms_feat.append(mapped_layer)
            elif l == self.out_layers - 1:
                # Downsample the last two layers, concat them and then map their channels
                layer1, layer2 = ms_feat[-1], ms_feat[-2]
                layer1 = F.interpolate(layer1, scale_factor=0.5, mode="bilinear", align_corners=True, recompute_scale_factor=True)
                layer2 = F.interpolate(layer2, scale_factor=0.25, mode="bilinear", align_corners=True,recompute_scale_factor=True)
                mapped_layer = self.mapper_layer_list[l](torch.cat((layer1, layer2), dim=1))
                out_ms_feat.append(mapped_layer)
            else:
                mapped_layer = self.mapper_layer_list[l](ms_feat[ms_feat_idx])
                out_ms_feat.append(mapped_layer)
                ms_feat_idx += 1

        return out_ms_feat


class OpticalFlowHead(nn.Module):
    """Optical flow decoder head. Currently uses the HD3 decoder structure."""
    def __init__(self, corr_range, ds=6):
        super(OpticalFlowHead, self).__init__()
        self.corr_range = corr_range
        self.levels = len(self.corr_range)
        self.classes = [(2 * d + 1) ** 2 for d in corr_range]
        self.ds = ds  # Downsample ratio of the coarsest level

        self.feature_mapper = OpticalFlowMapper(in_channels_list=[512, 256, 256, 256, 256, 512],
                                                out_channels_list=[32, 64, 128, 256, 512, 512])

        self.decoder = HDADecoder
        self.dim = 2

        pyr_channels = [16, 32, 64, 128, 256, 512, 512]
        feat_d_offset = pyr_channels[::-1]
        feat_d_offset[0] = 0
        up_d_offset = [0] + self.classes[1:]
        for l in range(self.levels):
            setattr(self, 'cost_bn_{}'.format(l), nn.BatchNorm2d(self.classes[l]))

            input_d = self.classes[l] + feat_d_offset[l] + up_d_offset[l] + 2 * (l > 0)
            if l < self.levels - 1:
                up_classes = self.classes[l + 1]
            else:
                up_classes = -1

            setattr(self, 'Decoder_{}'.format(l), Decoder(input_d, self.decoder, self.classes[l], up_classes=up_classes))

    def warp(self, x, flow_vect):
        return flow_warp(x, flow_vect)

    # prev_feat are the multi-scale features of the previous image
    # curr_feat are the multi-scale features of the current image
    def forward(self, prev_ms_feat, curr_ms_feat):

        # Map the features from EfficientPS to the Optical Flow decoder
        prev_ms_feat = self.feature_mapper(prev_ms_feat)
        curr_ms_feat = self.feature_mapper(curr_ms_feat)

        prev_ms_feat = [f[:, :, :, :] for f in prev_ms_feat[::-1]]
        curr_ms_feat = [f[:, :, :, :] for f in curr_ms_feat[::-1]]

        # Output of the multi-scale features
        ms_pred = []
        up_curr_level_flow = None
        for l in range(self.levels):
            prev_level = prev_ms_feat[l]
            curr_level = curr_ms_feat[l]

            if l == 0:
                curr_level_corr = curr_level
            else:
                # print("Curr_level", curr_level.shape)
                # print("Up_curr_level", up_curr_level_flow.shape)
                curr_level_corr = self.warp(curr_level, up_curr_level_flow)

            cost_vol = correlation.FunctionCorrelation(tensorFirst=prev_level, tensorSecond=curr_level_corr)
            cost_bn = getattr(self, "cost_bn_{}".format(l))
            cost_vol = cost_bn(cost_vol)

            if l == 0:
                decoder_input = cost_vol
            else:
                decoder_input = torch.cat([cost_vol, prev_level, ms_pred[-1][-1], up_curr_level_flow], dim=1)

            decoder = getattr(self, "Decoder_{}".format(l))
            prob_map, up_level_flow = decoder(decoder_input)

            curr_level_flow = density2vector(prob_map, True)
            if l > 0:
                curr_level_flow += up_curr_level_flow
            ms_pred.append([prob_map, curr_level_flow * 2**(self.ds - l), up_level_flow])

            if l < self.levels - 1:
                up_curr_level_flow = 2 * F.interpolate(curr_level_flow, scale_factor=2, mode="bilinear", align_corners=True, recompute_scale_factor=True)

        ms_prob = [l[0] for l in ms_pred]
        ms_flow = [l[1] for l in ms_pred]

        return ms_prob, ms_flow


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, activation=True):
    if activation:
        return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=True),
                             nn.LeakyReLU(0.1))
    else:
        return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=True))


def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding, bias=True)


def deformable_conv(in_planes, out_planes, kernel_size=3, strides=1, padding=1, use_bias=True):
    return ops.DeformConv2d(in_planes, out_planes, kernel_size, strides, padding, bias=use_bias)
    # return nn.Conv2d(in_planes, out_planes, kernel_size, strides, padding, bias=use_bias)


def predict_flow(in_planes):
    return nn.Conv2d(in_planes, 2, kernel_size=3, stride=1, padding=1, bias=True)


def predict_mask(in_planes):
    return nn.Conv2d(in_planes, 1, kernel_size=3, stride=1, padding=1, bias=True)


class FlowDPCSmall(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FlowDPCSmall, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=2 * out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=2 * out_channels, out_channels=2 * out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(in_channels=2 * out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn3 = nn.BatchNorm2d(num_features=out_channels)
        self.elu = nn.ELU(inplace=False)

    def forward(self, x):
        x = self.elu(self.conv1(x))
        x = self.elu(self.conv2(x))
        x = self.elu(self.bn3(self.conv3(x)))
        return x

class ScaleFeatureMap(nn.Module):
    def __init__(self, in_channels, out_channels, scale_ratio=2, mode="nearest"):
        super(ScaleFeatureMap, self).__init__()
        self.elu = nn.ELU()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=2 * out_channels, bias=False, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=2 * out_channels, out_channels=out_channels, bias=False, kernel_size=3, stride=1, padding=1)
        self.scale_ratio = scale_ratio
        self.mode = mode

    def forward(self, x):
        scaled_x = F.interpolate(x, scale_factor=self.scale_ratio, mode=self.mode)
        conv1 = self.elu(self.conv1(scaled_x))
        out = self.elu(self.conv2(conv1))
        return out

class MultiScaleMerge(nn.Module):
    def __init__(self, ms_in_channels, int_channels):
        super(MultiScaleMerge, self).__init__()

        self.elu = nn.ELU(inplace=False)

        self.h32_dpc = FlowDPCSmall(ms_in_channels[-1], int_channels)
        self.h32x16 = ScaleFeatureMap(int_channels, int_channels, scale_ratio=2)
        self.h16_dpc = FlowDPCSmall(ms_in_channels[-2], int_channels)
        self.h16x8 = ScaleFeatureMap(int_channels, int_channels, scale_ratio=2)
        self.h8_dpc = FlowDPCSmall(ms_in_channels[-3], int_channels)
        self.h8x4 = ScaleFeatureMap(int_channels, int_channels, scale_ratio=2)
        self.h4_dpc = FlowDPCSmall(ms_in_channels[-4], int_channels)

        self.h32x4 = ScaleFeatureMap(int_channels, int_channels, scale_ratio=8)
        self.h16x4 = ScaleFeatureMap(int_channels, int_channels, scale_ratio=4)
        self.h8x4 = ScaleFeatureMap(int_channels, int_channels, scale_ratio=2)
        self.h4x2 = ScaleFeatureMap(int_channels, int_channels, scale_ratio=2)
        self.h2x1 = ScaleFeatureMap(int_channels, int_channels, scale_ratio=2)

        self.h16_sum_conv = nn.Conv2d(int_channels, int_channels, 3, 1, 1, bias=False)
        self.h8_sum_conv = nn.Conv2d(int_channels, int_channels, 3, 1, 1, bias=False)
        self.h4_sum_conv = nn.Conv2d(int_channels, int_channels, 3, 1, 1, bias=False)

        self.h4_cat_conv = nn.Conv2d(4 * int_channels, int_channels, 3, 1, 1, bias=False)

        self.make3_ch = nn.Conv2d(int_channels, 3, 1, 1, 0, bias=False)

    def forward(self, ms_feat):
        h32_dpc = self.h32_dpc(ms_feat[-1])
        h16_dpc = self.h16_dpc(ms_feat[-2])
        h8_dpc = self.h8_dpc(ms_feat[-3])
        h4_dpc = self.h4_dpc(ms_feat[-4])

        h32x16_int = self.h32x16(h32_dpc)
        h16x8_int = self.h16x8(h16_dpc)
        h8x4_int = self.h8x4(h8_dpc)
        h4x2_int = self.h4x2(h4_dpc)
        h2x1_int = self.h2x1(h4x2_int)

        h16_sum = self.elu(self.h16_sum_conv(torch.add(h32x16_int, h16_dpc)))
        h8_sum = self.elu(self.h8_sum_conv(torch.add(h16x8_int, h8_dpc)))
        h4_sum = self.elu(self.h4_sum_conv(torch.add(h8x4_int, h4_dpc)))

        h32x4 = self.h32x4(h32_dpc)
        h16x4 = self.h16x4(h16_sum)
        h8x4 = self.h8x4(h8_sum)
        h4x4 = h4_sum

        # Concat these 4 scales
        h4_cat = torch.cat([h32x4, h16x4, h8x4, h4x4], dim=1)
        h4_cat = self.elu(self.h4_cat_conv(h4_cat))
        h4_cat = self.make3_ch(h4_cat)

        return h4_cat


class OpticalFlowHeadMFN_S(nn.Module):
    def __init__(self, ms_in_channels, max_displacement=4, flow_multiplier=1., deform_bias=True, upfeat_ch=[16, 16, 16, 16], only_mfns=False):
        super(OpticalFlowHeadMFN_S, self).__init__()
        self.scale = 20. * flow_multiplier
        self.md = max_displacement
        self.deform_bias = deform_bias
        self.upfeat_ch = upfeat_ch
        self.strides = [32, 16, 8, 4]
        self.only_mnfs = only_mfns

        self.corr = SpatialCorrelationSampler(kernel_size=1, patch_size=(int(1 + 2 * self.md), int(1 + 2 * self.md)), stride=1,
                                              padding=0, dilation_patch=1)
        # self.corr = Correlation(pad_size=self.md, kernel_size=1, max_displacement=self.md, stride1=1, stride2=1, corr_multiply=1)
        self.leakyRELU = nn.LeakyReLU(0.1)

        nd = (2 * self.md + 1) ** 2
        dd = np.cumsum([128, 128, 96, 64, 32])

        od = nd
        self.conv32_0 = conv(od, 128, kernel_size=3, stride=1)
        self.conv32_1 = conv(od + dd[0], 128, kernel_size=3, stride=1)
        self.conv32_2 = conv(od + dd[1], 96, kernel_size=3, stride=1)
        self.conv32_3 = conv(od + dd[2], 64, kernel_size=3, stride=1)
        self.conv32_4 = conv(od + dd[3], 32, kernel_size=3, stride=1)
        self.pred_flow32 = predict_flow(od + dd[4])
        self.pred_mask32 = predict_mask(od + dd[4])
        self.upfeat32x16 = deconv(od + dd[4], self.upfeat_ch[0], kernel_size=4, stride=2, padding=1)

        od = nd + ms_in_channels[-2] + 18
        self.conv16_0 = conv(od, 128, kernel_size=3, stride=1)
        self.conv16_1 = conv(od + dd[0], 128, kernel_size=3, stride=1)
        self.conv16_2 = conv(od + dd[1], 96, kernel_size=3, stride=1)
        self.conv16_3 = conv(od + dd[2], 64, kernel_size=3, stride=1)
        self.conv16_4 = conv(od + dd[3], 32, kernel_size=3, stride=1)
        self.pred_flow16 = predict_flow(od + dd[4])
        self.pred_mask16 = predict_mask(od + dd[4])
        self.upfeat16x8 = deconv(od + dd[4], self.upfeat_ch[1], kernel_size=4, stride=2, padding=1)

        od = nd + ms_in_channels[-3] + 18  # nd + ms_in_channels[] + feat + flow
        self.conv8_0 = conv(od, 128, kernel_size=3, stride=1)
        self.conv8_1 = conv(od + dd[0], 128, kernel_size=3, stride=1)
        self.conv8_2 = conv(od + dd[1], 96, kernel_size=3, stride=1)
        self.conv8_3 = conv(od + dd[2], 64, kernel_size=3, stride=1)
        self.conv8_4 = conv(od + dd[3], 32, kernel_size=3, stride=1)
        self.pred_flow8 = predict_flow(od + dd[4])
        self.pred_mask8 = predict_mask(od + dd[4])
        self.upfeat8x4 = deconv(od + dd[4], self.upfeat_ch[2], kernel_size=4, stride=2, padding=1)

        od = nd + ms_in_channels[-4] + 18
        self.conv4_0 = conv(od, 128, kernel_size=3, stride=1)
        self.conv4_1 = conv(od + dd[0], 128, kernel_size=3, stride=1)
        self.conv4_2 = conv(od + dd[1], 96, kernel_size=3, stride=1)
        self.conv4_3 = conv(od + dd[2], 64, kernel_size=3, stride=1)
        self.conv4_4 = conv(od + dd[3], 32, kernel_size=3, stride=1)
        self.pred_flow4 = predict_flow(od + dd[4])

        self.dc_conv1 = conv(od + dd[4], 128, kernel_size=3, stride=1, padding=1, dilation=1)
        self.dc_conv2 = conv(128, 128, kernel_size=3, stride=1, padding=2, dilation=2)
        self.dc_conv3 = conv(128, 128, kernel_size=3, stride=1, padding=4, dilation=4)
        self.dc_conv4 = conv(128, 96, kernel_size=3, stride=1, padding=8, dilation=8)
        self.dc_conv5 = conv(96, 64, kernel_size=3, stride=1, padding=16, dilation=16)
        self.dc_conv6 = conv(64, 32, kernel_size=3, stride=1, padding=1, dilation=1)
        self.dc_conv7 = predict_flow(32)

        self.deform16 = deformable_conv(256, 256)
        self.deform8 = deformable_conv(256, 256)
        self.deform4 = deformable_conv(256, 256)

        self.conv16f = conv(16, 256, kernel_size=3, stride=1, padding=1, activation=False)
        self.conv8f = conv(16, 256, kernel_size=3, stride=1, padding=1, activation=False)
        self.conv4f = conv(16, 256, kernel_size=3, stride=1, padding=1, activation=False)

        self.multi_scale_merge = MultiScaleMerge(ms_in_channels, 64)

    def forward(self, ms_feat_prev, ms_feat_curr):

        # Coarsest feature map (H/32)
        # Compute the correlation
        corr32 = self.perform_correlation(ms_feat_prev[-1], ms_feat_curr[-1])  # od
        corr32 = self.leakyRELU(corr32)  # od

        # Compute the occlusion mask (theta), trade off term (mu), and flow displacement (phi)
        # Dense convolutions
        x = torch.cat([self.conv32_0(corr32), corr32], dim=1)  # x = 128 + (od)
        x = torch.cat([self.conv32_1(x), x], dim=1)  # x = 128 + (128+od)
        x = torch.cat([self.conv32_2(x), x], dim=1)  # 96 + (od+128+128)
        x = torch.cat([self.conv32_3(x), x], dim=1)  # 64 + (od+128+128+96)
        x = torch.cat([self.conv32_4(x), x], dim=1)  # 32 + (od+128+128+96+64)
        flow32 = self.pred_flow32(x)  # 2
        mask32 = self.pred_mask32(x)  # 1

        # H/16 feature map
        feat16 = self.leakyRELU(self.upfeat32x16(x))  # 16 (upfeat_ch[0])
        flow16 = Upsample(flow32, 2)  # 2
        mask16 = Upsample(mask32, 2)  # 1
        warp16 = (flow16 * self.scale / self.strides[1]).unsqueeze(1)
        warp16 = torch.repeat_interleave(warp16, 9, 1)
        S1, S2, S3, S4, S5 = warp16.shape
        warp16 = warp16.clone().view(S1, S2 * S3, S4, S5)
        warp16 = self.deform16(ms_feat_curr[-2], warp16)  # 256
        tradeoff16 = feat16
        warp16 = (warp16 * torch.sigmoid(mask16)) + self.conv16f(tradeoff16)
        warp16 = self.leakyRELU(warp16)
        corr16 = self.perform_correlation(ms_feat_prev[-2], warp16)
        corr16 = self.leakyRELU(corr16)
        x = torch.cat([corr16, ms_feat_prev[-2], feat16, flow16], dim=1)
        x = torch.cat([self.conv16_0(x), x], dim=1)
        x = torch.cat([self.conv16_1(x), x], dim=1)
        x = torch.cat([self.conv16_2(x), x], dim=1)
        x = torch.cat([self.conv16_3(x), x], dim=1)
        x = torch.cat([self.conv16_4(x), x], dim=1)
        flow16 = flow16 + self.pred_flow16(x)
        mask16 = self.pred_mask16(x)

        # H/8 feature map
        feat8 = self.leakyRELU(self.upfeat16x8(x))
        flow8 = Upsample(flow16, 2)
        mask8 = Upsample(mask16, 2)
        warp8 = (flow8 * self.scale / self.strides[2]).unsqueeze(1)
        warp8 = torch.repeat_interleave(warp8, 9, 1)
        # warp8 = torch.cat([warp8] * 9, dim=1)
        S1, S2, S3, S4, S5 = warp8.shape
        warp8 = warp8.clone().view(S1, S2 * S3, S4, S5)
        warp8 = self.deform8(ms_feat_curr[-3], warp8)
        tradeoff8 = feat8
        warp8 = (warp8 * torch.sigmoid(mask8)) + self.conv8f(tradeoff8)
        warp8 = self.leakyRELU(warp8)
        corr8 = self.perform_correlation(ms_feat_prev[-3], warp8)
        corr8 = self.leakyRELU(corr8)
        x = torch.cat([corr8, ms_feat_prev[-3], feat8, flow8], dim=1)  # 355
        x = torch.cat([self.conv8_0(x), x], dim=1)  #
        x = torch.cat([self.conv8_1(x), x], dim=1)
        x = torch.cat([self.conv8_2(x), x], dim=1)
        x = torch.cat([self.conv8_3(x), x], dim=1)
        x = torch.cat([self.conv8_4(x), x], dim=1)
        flow8 = flow8 + self.pred_flow8(x)
        mask8 = self.pred_mask8(x)

        # H/4 feature map
        feat4 = self.leakyRELU(self.upfeat8x4(x))
        flow4 = Upsample(flow8, 2)
        mask4 = Upsample(mask8, 2)
        warp4 = (flow4 * self.scale / self.strides[3]).unsqueeze(1)
        warp4 = torch.repeat_interleave(warp4, 9, 1)
        S1, S2, S3, S4, S5 = warp4.shape
        warp4 = warp4.clone().view(S1, S2 * S3, S4, S5)
        warp4 = self.deform4(ms_feat_curr[-4], warp4)
        tradeoff4 = feat4
        warp4 = (warp4 * torch.sigmoid(mask4)) + self.conv4f(tradeoff4)
        warp4 = self.leakyRELU(warp4)
        corr4 = self.perform_correlation(ms_feat_prev[-4], warp4)
        corr4 = self.leakyRELU(corr4)
        x = torch.cat([corr4, ms_feat_prev[-4], feat4, flow4], dim=1)
        x = torch.cat([self.conv4_0(x), x], dim=1)
        x = torch.cat([self.conv4_1(x), x], dim=1)
        x = torch.cat([self.conv4_2(x), x], dim=1)
        x = torch.cat([self.conv4_3(x), x], dim=1)
        x = torch.cat([self.conv4_4(x), x], dim=1)
        flow4 = flow4 + self.pred_flow4(x)

        x = self.dc_conv4(self.dc_conv3(self.dc_conv2(self.dc_conv1(x))))
        flow4 = flow4 + self.dc_conv7(self.dc_conv6(self.dc_conv5(x)))

        if self.only_mnfs:
            preds = [(flow * self.scale) for flow in [flow4, flow8, flow16, flow32]]
            visuals = Upsample(flow4, 4)
            return preds, visuals

        predictions = [flow * self.scale for flow in [flow4, flow8, flow16, flow32]]
        occlusion_masks = []
        occlusion_masks.append(torch.sigmoid(mask4))
        flows = [flow4, flow8, flow16, flow32]
        mask1 = Upsample(mask4, 4)
        mask1 = torch.sigmoid(mask1) - 0.5

        # Merge all scales into one scale
        ms_feat_prev_merge = self.multi_scale_merge(ms_feat_prev)
        ms_feat_curr_merge = self.multi_scale_merge(ms_feat_curr)

        ch30 = Upsample(ms_feat_prev_merge, 4)
        ch30 = torch.cat([ch30, torch.zeros_like(mask1)], dim=1)
        ch40 = self.warp(Upsample(ms_feat_curr_merge, 4), Upsample(flow4, 4) * self.scale)
        ch40 = torch.cat([ch40, mask1], dim=1)

        srcs = [ms_feat_prev, ms_feat_curr, flows, ch30, ch40]
        return predictions, occlusion_masks, srcs

    def perform_correlation(self, feat1, feat2):
        b, c, h, w = feat1.shape
        corr = self.corr(feat1.contiguous(), feat2.contiguous())
        corr = corr / c
        corr = corr.view(b, -1, h, w)
        return corr

    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow
        """
        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()

        device = x.device
        grid = grid.to(device)
        # vgrid = Variable(grid) + flo
        vgrid = grid + torch.flip(flo, [1])

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0*vgrid[:, 0, :, :].clone() / max(W-1, 1)-1.0
        vgrid[:, 1, :, :] = 2.0*vgrid[:, 1, :, :].clone() / max(H-1, 1)-1.0

        vgrid = vgrid.permute(0, 2, 3, 1)
        # vgrid = vgrid.permute(0,2,3,1).clamp(-1.1, 1.1)
        output = nn.functional.grid_sample(x, vgrid, align_corners=True)
        mask = torch.ones(x.size()).to(device)
        mask = nn.functional.grid_sample(mask, vgrid, align_corners=True)

        # if W==128:
        # np.save('mask.npy', mask.cpu().data.numpy())
        # np.save('warp.npy', output.cpu().data.numpy())

        mask[mask < 0.9999] = 0
        mask[mask > 0] = 1

        return output * mask


class OpticalFlowHeadMFN(nn.Module):
    def __init__(self, ms_in_channels, flow_multiplier=1., deform_bias=True, upfeat_ch=[16, 16, 16, 16], **kwargs):
        super(OpticalFlowHeadMFN, self).__init__()
        self.strides = [32, 16, 8, 4]
        self.md = 2
        self.scale = 20. * flow_multiplier
        self.deform_bias = deform_bias
        self.upfeat_ch = upfeat_ch
        self.ms_in_channels = ms_in_channels

        self.mfn_s = OpticalFlowHeadMFN_S(ms_in_channels, 4, flow_multiplier, deform_bias, upfeat_ch)
        self.leakyRELU = nn.LeakyReLU(0.1)

        self.conv1x = conv(4, 16, stride=2)
        self.conv1y = conv(16, 16, stride=1)
        self.conv1z = conv(16, 16, stride=1)
        self.conv2x = conv(16, 32, stride=2)
        self.conv2y = conv(32, 32, stride=1)
        self.conv2z = conv(32, 32, stride=1)
        self.conv3x = conv(32, 64, stride=2)
        self.conv3y = conv(64, 64, stride=1)
        self.conv3z = conv(64, 64, stride=1)
        self.conv4x = conv(64, 96, stride=2)
        self.conv4y = conv(96, 96, stride=1)
        self.conv4z = conv(96, 96, stride=1)
        self.conv5x = conv(96, 128, stride=2)
        self.conv5y = conv(128, 128, stride=1)
        self.conv5z = conv(128, 128, stride=1)
        self.conv6x = conv(128, 196, stride=2)
        self.conv6y = conv(196, 196, stride=1)
        self.conv6z = conv(196, 196, stride=1)

        self.corr = SpatialCorrelationSampler(kernel_size=1, patch_size=(int(1 + 2 * self.md), int(1 + 2 * self.md)), stride=1,
                                              padding=0, dilation_patch=1)
        # self.corr = Correlation(pad_size=self.md, kernel_size=1, max_displacement=self.md, stride1=1, stride2=1, corr_multiply=1)

        nd = (2 * self.md + 1) ** 2
        dd = np.cumsum([128, 128, 96, 64, 32])

        od = nd + nd + 2
        self.conv32_0 = conv(od, 128, kernel_size=3, stride=1)
        self.conv32_1 = conv(od + dd[0], 128, kernel_size=3, stride=1)
        self.conv32_2 = conv(od + dd[1], 96, kernel_size=3, stride=1)
        self.conv32_3 = conv(od + dd[2], 64, kernel_size=3, stride=1)
        self.conv32_4 = conv(od + dd[3], 32, kernel_size=3, stride=1)
        self.pred_flow32 = predict_flow(od + dd[4])
        self.upfeat32x16 = deconv(od + dd[4], self.upfeat_ch[0], kernel_size=4, stride=2, padding=1)

        od = nd + nd + 256 + 16 + 2 + 2
        self.conv16_0 = conv(od, 128, kernel_size=3, stride=1)
        self.conv16_1 = conv(od + dd[0], 128, kernel_size=3, stride=1)
        self.conv16_2 = conv(od + dd[1], 96, kernel_size=3, stride=1)
        self.conv16_3 = conv(od + dd[2], 64, kernel_size=3, stride=1)
        self.conv16_4 = conv(od + dd[3], 32, kernel_size=3, stride=1)
        self.pred_flow16 = predict_flow(od + dd[4])
        self.upfeat16x8 = deconv(od + dd[4], self.upfeat_ch[1], kernel_size=4, stride=2, padding=1)

        od = nd + nd + 256 + 16 + 2 + 2
        self.conv8_0 = conv(od, 128, kernel_size=3, stride=1)
        self.conv8_1 = conv(od + dd[0], 128, kernel_size=3, stride=1)
        self.conv8_2 = conv(od + dd[1], 96, kernel_size=3, stride=1)
        self.conv8_3 = conv(od + dd[2], 64, kernel_size=3, stride=1)
        self.conv8_4 = conv(od + dd[3], 32, kernel_size=3, stride=1)
        self.pred_flow8 = predict_flow(od + dd[4])
        self.upfeat8x4 = deconv(od + dd[4], self.upfeat_ch[2], kernel_size=4, stride=2, padding=1)

        od = nd + nd + 256 + 16 + 2 + 2
        self.conv4_0 = conv(od, 128, kernel_size=3, stride=1)
        self.conv4_1 = conv(od + dd[0], 128, kernel_size=3, stride=1)
        self.conv4_2 = conv(od + dd[1], 96, kernel_size=3, stride=1)
        self.conv4_3 = conv(od + dd[2], 64, kernel_size=3, stride=1)
        self.conv4_4 = conv(od + dd[3], 32, kernel_size=3, stride=1)
        self.pred_flow4 = predict_flow(od + dd[4])

        self.dc_conv1 = conv(od + dd[4], 128, kernel_size=3, stride=1, padding=1, dilation=1)
        self.dc_conv2 = conv(128, 128, kernel_size=3, stride=1, padding=2, dilation=2)
        self.dc_conv3 = conv(128, 128, kernel_size=3, stride=1, padding=4, dilation=4)
        self.dc_conv4 = conv(128, 96, kernel_size=3, stride=1, padding=8, dilation=8)
        self.dc_conv5 = conv(96, 64, kernel_size=3, stride=1, padding=16, dilation=16)
        self.dc_conv6 = conv(64, 32, kernel_size=3, stride=1, padding=1, dilation=1)
        self.dc_conv7 = predict_flow(32)

        self.deform32 = deformable_conv(256, 256)
        self.deform16 = deformable_conv(256, 256)
        self.deform8 = deformable_conv(256, 256)
        self.deform4 = deformable_conv(256, 256)

    def forward(self, ms_feat_prev, ms_feat_curr, img_prev, img_curr):
        _, _, srcs = self.mfn_s(ms_feat_prev, ms_feat_curr)
        ms_feat_prev, ms_feat_curr, flows, ch30, ch40 = srcs

        mfn_feat_prev2 = self.conv1z(self.conv1y(self.conv1x(ch30)))
        mfn_feat_prev4 = self.conv2z(self.conv2y(self.conv2x(mfn_feat_prev2)))
        mfn_feat_prev8 = self.conv3z(self.conv3y(self.conv3x(mfn_feat_prev4)))
        mfn_feat_prev16 = self.conv4z(self.conv4y(self.conv4x(mfn_feat_prev8)))
        mfn_feat_prev32 = self.conv5z(self.conv5y(self.conv5x(mfn_feat_prev16)))
        # mfn_feat_prev32 = self.conv6z(self.conv6y(self.conv6x(mfn_feat_prev16)))
        mfn_feat_prev = [mfn_feat_prev4, mfn_feat_prev8, mfn_feat_prev16, mfn_feat_prev32]

        mfn_feat_curr2 = self.conv1z(self.conv1y(self.conv1x(ch40)))
        mfn_feat_curr4 = self.conv2z(self.conv2y(self.conv2x(mfn_feat_curr2)))
        mfn_feat_curr8 = self.conv3z(self.conv3y(self.conv3x(mfn_feat_curr4)))
        mfn_feat_curr16 = self.conv4z(self.conv4y(self.conv4x(mfn_feat_curr8)))
        mfn_feat_curr32 = self.conv5z(self.conv5y(self.conv5x(mfn_feat_curr16)))
        # mfn_feat_curr32 = self.conv6z(self.conv6y(self.conv6x(mfn_feat_curr16)))
        mfn_feat_curr = [mfn_feat_curr4, mfn_feat_curr8, mfn_feat_curr16, mfn_feat_curr32]

        flow32 = flows[-1]
        warp32u = (flow32 * self.scale / self.strides[0]).unsqueeze(1)
        warp32u = torch.repeat_interleave(warp32u, 9, 1)
        S1, S2, S3, S4, S5 = warp32u.shape
        warp32u = warp32u.clone().view(S1, S2*S3, S4, S5)
        warp32u = self.deform32(ms_feat_curr[-1], warp32u)
        warp32u = self.leakyRELU(warp32u)
        corr32u = self.leakyRELU(self.perform_correlation(ms_feat_prev[-1], warp32u))
        warp32v = mfn_feat_curr[-1]
        corr32v = self.leakyRELU(self.perform_correlation(mfn_feat_prev[-1], warp32v))
        x = torch.cat([corr32u, corr32v, flow32], dim=1)
        x = torch.cat([self.conv32_0(x), x], dim=1)
        x = torch.cat([self.conv32_1(x), x], dim=1)
        x = torch.cat([self.conv32_2(x), x], dim=1)
        x = torch.cat([self.conv32_3(x), x], dim=1)
        x = torch.cat([self.conv32_4(x), x], dim=1)
        flow32 = flow32 + self.pred_flow32(x)

        feat16 = self.leakyRELU(self.upfeat32x16(x))
        flow16 = Upsample(flow32, 2)
        warp16u = (flow16 * self.scale / self.strides[1]).unsqueeze(1)
        warp16u = torch.repeat_interleave(warp16u, 9, 1)
        S1, S2, S3, S4, S5 = warp16u.shape
        warp16u = warp16u.clone().view(S1, S2*S3, S4, S5)
        warp16u = self.deform16(ms_feat_curr[-2], warp16u)
        warp16u = self.leakyRELU(warp16u)
        corr16u = self.leakyRELU(self.perform_correlation(ms_feat_prev[-2], warp16u))
        warp16v = mfn_feat_curr[-2]
        corr16v = self.leakyRELU(self.perform_correlation(mfn_feat_prev[-2], warp16v))
        # print(ms_feat_prev[-2].shape)
        # print(feat16.shape)
        # print(corr16u.shape)
        # print(corr16v.shape)
        # print(flow16.shape)
        # print(flows[-2].shape)
        x = torch.cat([ms_feat_prev[-2], feat16, corr16u, corr16v, flow16, flows[-2]], dim=1)
        x = torch.cat([self.conv16_0(x), x], dim=1)
        x = torch.cat([self.conv16_1(x), x], dim=1)
        x = torch.cat([self.conv16_2(x), x], dim=1)
        x = torch.cat([self.conv16_3(x), x], dim=1)
        x = torch.cat([self.conv16_4(x), x], dim=1)
        flow16 = flow16 + self.pred_flow16(x)

        feat8 = self.leakyRELU(self.upfeat16x8(x))
        flow8 = Upsample(flow16, 2)
        warp8u = (flow8 * self.scale / self.strides[2]).unsqueeze(1)
        warp8u = torch.repeat_interleave(warp8u, 9, 1)
        # warp8u = torch.cat([warp8u] * 9, dim=1)
        # print(warp8u.shape)
        S1, S2, S3, S4, S5 = warp8u.shape
        warp8u = warp8u.clone().view(S1, S2 * S3, S4, S5)
        warp8u = self.deform8(ms_feat_curr[-3], warp8u)
        warp8u = self.leakyRELU(warp8u)
        corr8u = self.leakyRELU(self.perform_correlation(ms_feat_prev[-3], warp8u))
        warp8v = mfn_feat_curr[-3]
        corr8v = self.leakyRELU(self.perform_correlation(mfn_feat_prev[-3], warp8v))
        x = torch.cat([ms_feat_prev[-3], feat8, corr8u, corr8v, flow8, flows[-3]], dim=1)
        x = torch.cat([self.conv8_0(x), x], dim=1)
        x = torch.cat([self.conv8_1(x), x], dim=1)
        x = torch.cat([self.conv8_2(x), x], dim=1)
        x = torch.cat([self.conv8_3(x), x], dim=1)
        x = torch.cat([self.conv8_4(x), x], dim=1)
        flow8 = flow8 + self.pred_flow8(x)

        feat4 = self.leakyRELU(self.upfeat8x4(x))
        flow4 = Upsample(flow8, 2)
        warp4u = (flow4 * self.scale / self.strides[3]).unsqueeze(1)
        warp4u = torch.repeat_interleave(warp4u, 9, 1)
        S1, S2, S3, S4, S5 = warp4u.shape
        warp4u = warp4u.clone().view(S1, S2 * S3, S4, S5)
        warp4u = self.deform4(ms_feat_curr[-4], warp4u)
        warp4u = self.leakyRELU(warp4u)
        corr4u = self.leakyRELU(self.perform_correlation(ms_feat_prev[-4], warp4u))
        warp4v = mfn_feat_curr[-4]
        corr4v = self.leakyRELU(self.perform_correlation(mfn_feat_prev[-4], warp4v))
        x = torch.cat([ms_feat_prev[-4], feat4, corr4u, corr4v, flow4, flows[-4]], dim=1)
        x = torch.cat([self.conv4_0(x), x], dim=1)
        x = torch.cat([self.conv4_1(x), x], dim=1)
        x = torch.cat([self.conv4_2(x), x], dim=1)
        x = torch.cat([self.conv4_3(x), x], dim=1)
        x = torch.cat([self.conv4_4(x), x], dim=1)
        flow4 = flow4 + self.pred_flow4(x)

        x = self.dc_conv4(self.dc_conv3(self.dc_conv2(self.dc_conv1(x))))
        flow4 = flow4 + self.dc_conv7(self.dc_conv6(self.dc_conv5(x)))

        preds = [(flow * self.scale) for flow in [flow4, flow8, flow16, flow32]]
        visuals = Upsample(flow4, 4)
        return preds, visuals

    def perform_correlation(self, feat1, feat2):
        b, c, h, w = feat1.shape
        corr = self.corr(feat1.contiguous(), feat2.contiguous())
        corr = corr / c
        corr = corr.view(b, -1, h, w)
        return corr

    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow
        """
        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), dim=1).float()

        device = x.device
        grid = grid.to(device)
        vgrid = grid + torch.flip(flo, [1])

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W-1, 1)-1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H-1, 1)-1.0

        # vgrid = vgrid.permute(0,2,3,1)
        vgrid = vgrid.permute(0, 2, 3, 1).clamp(-1.1, 1.1)
        output = nn.functional.grid_sample(x, vgrid, align_corners=True)
        mask = (torch.ones(x.size())).to(device)
        mask = nn.functional.grid_sample(mask, vgrid, align_corners=True)

        # if W==128:
        # np.save('mask.npy', mask.cpu().data.numpy())
        # np.save('warp.npy', output.cpu().data.numpy())

        mask[mask < 0.9999] = 0
        mask[mask > 0] = 1

        return output * mask


class OpticalFlowHeadMFNInstance(nn.Module):
    def __init__(self, ms_in_channels, flow_multiplier=1., deform_bias=True, upfeat_ch=[16, 16, 16, 16], **kwargs):
        super(OpticalFlowHeadMFNInstance, self).__init__()

        # The MaskFlowNet network
        self.mfn = OpticalFlowHeadMFN(ms_in_channels, flow_multiplier, deform_bias, upfeat_ch, **kwargs)

        # Differentiable PnP module
        self.bpnp = BPnP.BPnP_m3d.apply

        self.THING_CLASSES = [11, 12, 13]

    def forward(self, ms_feat_prev, ms_feat_curr, img_prev, img_curr, flow_gt=None, points_2d=None, valid_2d=None, points_3d=None, valid_3d=None, points_flag=None, k=None):
        # Get the predictions from the MFN network
        flow_preds, visuals = self.mfn(ms_feat_prev, ms_feat_curr, img_prev, img_curr)

        if points_2d is None:
            # return flow_preds, visuals, {"pnp_gt": torch.tensor(0.).to(img_prev.device), "pnp_pred": torch.tensor(0.).to(img_prev.device)}
            return flow_preds, visuals, {"pose_error": torch.tensor(0.).to(img_prev.device)}

        # Do this only when the points_2d is not None
        # Scaling the pred and GT flow by a factor of 8
        # flow = F.interpolate(flow_preds[0], scale_factor=0.5, mode="bilinear", recompute_scale_factor=True, align_corners=True) * 0.125
        # flow_gt_small = F.interpolate(flow_gt, scale_factor=0.125, mode='bilinear', recompute_scale_factor=True, align_corners=True) * 0.125
        flow = flow_preds[0] * 0.25
        flow_gt_small = F.interpolate(flow_gt, scale_factor=0.25, mode='bilinear', recompute_scale_factor=True, align_corners=True) * 0.25


        # points_2d_pred_list = []
        # points_3d_valid_list = []
        # points_2d_gt_list = []
        max_point_count = 0
        total_inst_count = 0

        # Get the max number of points in this batch
        for b in range(flow.shape[0]):
            for inst_id in range(points_2d.shape[1]):
                if points_flag[b, inst_id]:
                    valid_indices = points_2d[b, inst_id, valid_2d[b, inst_id].squeeze()]
                    if valid_indices.shape[0] > max_point_count:
                        max_point_count = valid_indices.shape[0]
                    if valid_indices.shape[0] >= 20:
                        total_inst_count += 1

        if torch.sum(points_flag) == 0:
            return flow_preds, visuals, {"pose_error": torch.tensor(0.).to(img_prev.device)}

        points_2d_pred_tensor = torch.ones((total_inst_count, max_point_count, 2), dtype=torch.float32, requires_grad=True).to(img_prev.device) * -2000.
        points_2d_gt_tensor = torch.ones((total_inst_count, max_point_count, 2), dtype=torch.float32).to(img_prev.device) * -2000.
        points_3d_valid_tensor = torch.ones((total_inst_count, max_point_count, 3), dtype=torch.float32).to(img_prev.device) * -2000.
        points_mask_tensor = torch.zeros((total_inst_count), dtype=torch.bool).to(img_prev.device)  # Tells if the instance has at least than 20 points

        k_list = torch.zeros((total_inst_count, 3, 3), dtype=torch.float32).to(img_prev.device)
        curr_idx = 0
        for b in range(points_2d.shape[0]):
            for inst_id in range(points_2d.shape[1]):
                if points_flag[b, inst_id]:
                    valid_indices = points_2d[b, inst_id, valid_2d[b, inst_id].squeeze()]
                    valid_flow = flow[b, :, valid_indices[:, 1].long(), valid_indices[:, 0].long()].permute(1, 0)
                    gt_flow = flow_gt_small[b, :, valid_indices[:, 1].long(), valid_indices[:, 0].long()].permute(1, 0)
                    points_2d_pred = valid_indices + valid_flow
                    points_2d_gt = valid_indices + gt_flow

                    points_3d_valid = points_3d[b, inst_id, valid_3d[b, inst_id].squeeze()]

                    if points_2d_pred.shape[0] < 20:
                        continue

                    points_2d_pred_tensor[curr_idx, :points_2d_pred.shape[0], :] = points_2d_pred
                    points_2d_gt_tensor[curr_idx, :points_2d_gt.shape[0], :] = points_2d_gt
                    points_3d_valid_tensor[curr_idx, :points_3d_valid.shape[0], :] = points_3d_valid

                    points_mask_tensor[curr_idx] = True

                    # points_2d_pred_list.append(points_2d_pred)
                    # points_3d_valid_list.append(points_3d_valid)
                    # points_2d_gt_list.append(points_2d_gt)

                    # if valid_indices.shape[0] > max_point_count:
                    #     max_point_count = valid_indices.shape[0]

                    # # Sample 100 points from them
                    # if valid_indices.shape[0] > 100:
                    #     probs = (torch.ones((points_3d_valid.shape[0]), dtype=torch.float32) * 1. / points_3d_valid.shape[0])
                    #     sampled_idx = probs.multinomial(num_samples=100, replacement=False)
                    #
                    #     points_2d_pred = points_2d_pred[sampled_idx]
                    #     points_2d_gt = points_2d_gt[sampled_idx]
                    #     points_3d_valid = points_3d_valid[sampled_idx]
                    #
                    # points_2d_pred_tensor[curr_idx, :, :] = points_2d_pred
                    # points_2d_gt_tensor[curr_idx, :, :] = points_2d_gt
                    # points_3d_valid_tensor[curr_idx, :, :] = points_3d_valid
                    k_list[curr_idx, :, :] = k[b, :, :]
                    curr_idx += 1



        # points_2d_pred_tensor = self.insertPointsInTensor(points_2d_pred_list, points_2d_pred_tensor, max_point_count)
        # points_3d_valid_tensor = self.insertPointsInTensor(points_3d_valid_list, points_3d_valid_tensor, max_point_count)
        # points_2d_gt_tensor = self.insertPointsInTensor(points_2d_gt_list, points_2d_gt_tensor, max_point_count)
        #
        # pose_pred = self.bpnp(points_2d_pred_tensor, points_3d_valid_tensor, k[0, :, :])
        # pose_gt = self.bpnp(points_2d_gt_tensor, points_3d_valid_tensor, k[0, :, :])
        #
        # inst_coords_pnp_est_pred = BPnP.batch_project(pose_pred, points_3d_valid_tensor, k[0, :, :])
        # inst_coords_pnp_est_gt = BPnP.batch_project(pose_gt, points_3d_valid_tensor, k[0, :, :])
        #
        # mse_reproj_pnp_gt = self.computeProjectionMSE(inst_coords_pnp_est_pred, inst_coords_pnp_est_gt, 100)
        # mse_reproj_pnp_pred = self.computeProjectionMSE(inst_coords_pnp_est_pred, points_2d_pred_tensor, 100)

        hyps, scores = DSACStar.DSACStar.apply(points_2d_pred_tensor.cpu(), points_3d_valid_tensor.cpu(), k_list.cpu())
        hyps = hyps.to(img_prev.device)
        scores = scores.to(img_prev.device)
        # print("Scores", torch.sum(scores != scores), torch.sum(scores == float("Inf")))
        prob_scores = torch.softmax(scores, dim=1)

        hyps_gt, scores_gt = DSACStar.DSACStar.forward(None, points_2d_gt_tensor.cpu(), points_3d_valid_tensor.cpu(), k_list.cpu())
        hyps_gt = hyps_gt.to(img_prev.device)
        scores_gt = scores_gt.to(img_prev.device)
        prob_scores_gt = torch.softmax(scores_gt, dim=1)
        prob_scores_gt[prob_scores_gt != prob_scores_gt] *= 0.
        prob_max_indices = torch.argmax(prob_scores_gt, dim=1)

        if prob_max_indices.shape[0] == 1:
            print(prob_scores_gt.shape, torch.argmax(prob_scores_gt, dim=1).shape)
            pose_gt = hyps_gt[:, prob_max_indices.squeeze()]
            # pose_gt.to(img_prev.device)
            print(pose_gt.shape)
        else:
            pose_gt = torch.cat([hyps_gt_b[idx, :].unsqueeze(0) for hyps_gt_b, idx in zip(hyps_gt, torch.argmax(prob_scores_gt, dim=1).squeeze())], dim=0)
            # pose_gt = pose_gt.to(img_prev.device)

        pose_gt = torch.unsqueeze(pose_gt, dim=1)
        hyps_rot = kornia.angle_axis_to_quaternion(hyps[:, :, :3])
        gt_rot = kornia.angle_axis_to_quaternion(torch.cat(hyps.shape[1] * [pose_gt[:, :, :3]], dim=1))
        err_rot = torch.abs((gt_rot / torch.norm(gt_rot, dim=2).unsqueeze(2)) - (hyps_rot / torch.norm(hyps_rot, dim=2).unsqueeze(2)))

        # print("Rotation", torch.sum(err_rot != err_rot), torch.sum(err_rot == float("Inf")))

        hyps_trans = hyps[:, :, 3:]
        gt_trans = torch.cat(hyps.shape[1] * [pose_gt[:, :, 3:]], dim=1)
        err_trans = torch.abs(gt_trans - hyps_trans)

        # print("Translation", torch.sum(err_trans != err_trans), torch.sum(err_trans == float("Inf")))

        # Setting NaNs and Infs to 0.
        err_rot[err_rot == float("Inf")] = 0.
        err_rot[err_rot != err_rot] = 0.

        err_sum = torch.mean(err_rot, dim=2).unsqueeze(2) + torch.mean(err_trans, dim=2).unsqueeze(2)
        # print("Sum", torch.sum(err_sum != err_sum), torch.sum(err_sum == float("Inf")))

        point_count = torch.sum(points_mask_tensor) * err_sum.shape[1] * err_sum.shape[2]
        if point_count == 0:
            err_total = torch.sum(err_sum) * 0.
        else:
            # prob_scores_clone = prob_scores.detach().clone()
            # prob_scores_clone[prob_scores_clone != prob_scores_clone] = 0.
            point_mask = torch.cat(prob_scores.shape[1] * [points_mask_tensor.unsqueeze(1)], dim=1).unsqueeze(2)
            # err_total = torch.sum(point_mask * prob_scores * err_sum) / point_count
            err_total = torch.sum(point_mask * prob_scores * err_sum) / torch.sum(points_mask_tensor)

        # print("Total", err_total != err_total, err_total == float("Inf"))

        reproj_loss = {"pose_error": err_total}

        # reproj_loss = {"pnp_gt": mse_reproj_pnp_gt, "pnp_pred": mse_reproj_pnp_pred}

        # mse_reprojection_batch = ((inst_coords_pnp_est - points_2d_pred_list) ** 2).mean()
        # mse_reprojection_batch[mse_reprojection_batch > 100] = 100.

        # print(mse_reprojection_batch)

        # if depth_prev is not None:  # depth is None in inference
        #     for b in range(flow.shape[0]):
        #         # Compute the reprojection images for each instance
        #         img = img_prev[b, ...]
        #         depth = depth_prev[b, ...]
        #         inst_mask_full = inst_mask_full_prev[b, ...]
        #         camera_matrix = camera_matrix_prev[b, ...]
        #         cat_map = cat_map_prev[b, ...]
        #         flow_b = flow[b, ...]
        #
        #         # Get the list of instances from the instance mask
        #         inst_id_list = self.getUniqueInstanceIds(inst_mask_full, cat_map)
        #         mse_reprojection = self.computeReprojectionErrors(img, flow_b, depth, inst_mask_full, inst_id_list, camera_matrix)
        #
        #         mse_reprojection_batch[b] = mse_reprojection

        return flow_preds, visuals, reproj_loss

    def insertPointsInTensor(self, in_list, out_tensor, max_point_count):
        for t_idx in range(len(in_list)):
            point_diff = max_point_count
            start_idx = 0
            while point_diff > 0:
                pts_to_copy_count = min(in_list[t_idx].shape[0], point_diff)
                out_tensor[t_idx, start_idx:start_idx + pts_to_copy_count, :] = in_list[t_idx][:pts_to_copy_count, :]
                point_diff -= in_list[t_idx].shape[0]
                start_idx += in_list[t_idx].shape[0]

        return out_tensor

    def computeProjectionMSE(self, src, tgt, thresh):
        mse_reproj = torch.mean((src - tgt) ** 2, dim=[1, 2])
        large_loss_mask = mse_reproj <= thresh
        if torch.sum(large_loss_mask) <= 0:
            mse_reproj = torch.sum(large_loss_mask) * 0.
        else:
            mse_reproj = torch.sum(mse_reproj[large_loss_mask]) / torch.sum(large_loss_mask)

        return mse_reproj

    def computeReprojectionErrors(self, img, flow, depth, mask, inst_id_list, k):
        mse_inst_all = 0.
        for inst_id in inst_id_list.contiguous():
            inst_mask = (mask == inst_id)

            # Neglect instances smaller than 100 pixels
            if torch.sum(inst_mask) > 100:
                # Zero out all the elements apart from the instance
                # inst_img = img.clone()
                # inst_mask_ch = torch.cat([~inst_mask, ~inst_mask, ~inst_mask], dim=0)
                # inst_img[inst_mask_ch] = 0.

                # Zero out all the elements apart from the instance
                inst_flow = flow.clone()
                inst_mask_ch = torch.cat([~inst_mask, ~inst_mask], dim=0)
                inst_flow[inst_mask_ch] = False

                # Current instance indices
                inst_coords_curr = inst_mask.permute(1, 2, 0).squeeze().nonzero()

                # Get the 3D indices for the correponding pixels in the image
                inst_coords_curr_3d = torch.zeros((inst_coords_curr.shape[0], 3), dtype=torch.float32).to(img.device)
                depth_interpolate = self.interpolateDepth(depth, inst_coords_curr)
                for j in range(inst_coords_curr_3d.shape[0]):
                    row_curr = inst_coords_curr[j][0]
                    col_curr = inst_coords_curr[j][1]

                    inst_coords_curr_3d[j][0] = col_curr  # Column (X_w)
                    inst_coords_curr_3d[j][1] = row_curr  # Row (Y_w)
                    inst_coords_curr_3d[j][2] = depth_interpolate[j] / 100.
                    # inst_coords_curr_3d[j][2] = self.interpolateDepth(depth, row_curr, col_curr) / 100.

                # Predict the position of the instance in the next image using the current image and the predicted optical flow
                inst_coords_pred = torch.zeros_like(inst_coords_curr, dtype=torch.float32).to(img.device)  # This is be stored in X, Y format by default
                for j in range(inst_coords_pred.shape[0]):
                    flow_u = inst_flow[0, inst_coords_curr[j][0], inst_coords_curr[j][1]]  # x
                    flow_v = inst_flow[1, inst_coords_curr[j][0], inst_coords_curr[j][1]]  # y
                    inst_coords_pred[j][0] = inst_coords_curr[j][1] + flow_u  # Column (x)
                    inst_coords_pred[j][1] = inst_coords_curr[j][0] + flow_v  # Row (y)

                # Compute the R and t from the 2D-3D correspondences. The 3D from the current timestep and the 2D from the predicted mask image
                # We have a calibrated camera, so we can use the PnP algorithm
                inst_coords_pred = inst_coords_pred.unsqueeze(dim=0)

                if inst_coords_curr_3d.shape[0] > 50:
                    probs = (torch.ones((inst_coords_curr_3d.shape[0]), dtype=torch.float32) * 1. / inst_coords_curr_3d.shape[0]).to(img.device)
                    sampled_idx = probs.multinomial(num_samples=50, replacement=False)

                    inst_coords_curr_3d = inst_coords_curr_3d[sampled_idx]
                    inst_coords_pred = inst_coords_pred[:, sampled_idx]

                    del probs, sampled_idx

                pose = self.bpnp(inst_coords_pred, inst_coords_curr_3d, k)
                inst_coords_pnp_est = BPnP.batch_project(pose, inst_coords_curr_3d, k)

                # Compute the reprojection error
                mse_inst = ((inst_coords_pnp_est - inst_coords_pred) ** 2).mean()
                if mse_inst < 100:
                    mse_inst_all += mse_inst

                del inst_img, inst_mask_ch, inst_flow, inst_coords_curr, inst_coords_curr_3d, depth_interpolate, inst_coords_pred, pose, inst_coords_pnp_est

        del img, flow, depth, mask, inst_id_list, k

        return mse_inst_all

    def interpolateDepth(self, depth, coords):
        _, H, W = depth.shape
        depth_int = depth.unsqueeze(0)
        sample_coords_x = coords[:, 1].unsqueeze(0).unsqueeze(2).unsqueeze(3).float()
        sample_coords_y = coords[:, 0].unsqueeze(0).unsqueeze(2).unsqueeze(3).float()
        sample_coords = torch.cat([sample_coords_x, sample_coords_y], dim=3)

        # Normalise coords to [0, 1]
        sample_coords[:, :, :, 0] = sample_coords[:, :, :, 0] / (W - 1)
        sample_coords[:, :, :, 1] = sample_coords[:, :, :, 1] / (H - 1)

        # Normalise coords to [-1, 1]
        sample_coords = (sample_coords * 2) - 1

        # Interpolate the depth using the coordinates
        depth_interpolate = torch.nn.functional.grid_sample(depth_int, sample_coords, mode="bilinear",
                                                            padding_mode="border", align_corners=True)
        depth_interpolate = depth_interpolate.permute(2, 0, 1, 3).squeeze()
        return depth_interpolate

    def getUniqueInstanceIds(self, inst_mask, cat_map):
        # Get the list of unique instance IDs from the instance mask
        inst_id_list = torch.unique(inst_mask)

        # Get the boolean values as to whether an element is in the THING_CLASSES or not
        things_bool_list = [cat_map[inst_id_list] == d for d in self.THING_CLASSES]

        # Combine all the elements of THINGS_CLASSES using logical_or
        combined_things_bool = torch.logical_or(things_bool_list[0], things_bool_list[1])
        for i in range(2, len(things_bool_list)):
            combined_things_bool = torch.logical_or(combined_things_bool, things_bool_list[i])

        # Retain only the insts are that in THING_CLASSES
        # print(inst_id_list.shape, combined_things_bool.shape)
        inst_thing_list = inst_id_list[combined_things_bool]
        return inst_thing_list


class MultiScaleMergeRAFT(nn.Module):
    def __init__(self, ms_in_ch, inter_ch, out_ch):
        super(MultiScaleMergeRAFT, self).__init__()

        self.elu = nn.ELU(inplace=False)

        self.h32_dpc = FlowDPCSmall(ms_in_ch[-1], inter_ch)
        self.h32x16 = ScaleFeatureMap(inter_ch, inter_ch, scale_ratio=2)
        self.h16_dpc = FlowDPCSmall(ms_in_ch[-2], inter_ch)
        self.h16x8 = ScaleFeatureMap(inter_ch, inter_ch, scale_ratio=2)
        self.h8_dpc = FlowDPCSmall(ms_in_ch[-3], inter_ch)
        self.h8x4 = ScaleFeatureMap(inter_ch, inter_ch, scale_ratio=2)
        self.h4_dpc = FlowDPCSmall(ms_in_ch[-4], inter_ch)

        self.h32x4 = ScaleFeatureMap(inter_ch, inter_ch, scale_ratio=8)
        self.h16x4 = ScaleFeatureMap(inter_ch, inter_ch, scale_ratio=4)
        self.h8x4 = ScaleFeatureMap(inter_ch, inter_ch, scale_ratio=2)
        self.h4x2 = ScaleFeatureMap(inter_ch, inter_ch, scale_ratio=2)
        self.h2x1 = ScaleFeatureMap(inter_ch, inter_ch, scale_ratio=2)

        self.h16_sum_conv = nn.Conv2d(inter_ch, inter_ch, 3, 1, 1, bias=False)
        self.h8_sum_conv = nn.Conv2d(inter_ch, inter_ch, 3, 1, 1, bias=False)
        self.h4_sum_conv = nn.Conv2d(inter_ch, inter_ch, 3, 1, 1, bias=False)

        self.h4_cat_conv = nn.Conv2d(4 * inter_ch, inter_ch, 3, 1, 1, bias=False)

        self.make_out_ch = nn.Conv2d(inter_ch, out_ch, 1, 1, 0, bias=False)

    def forward(self, ms_feat):
        h32_dpc = self.h32_dpc(ms_feat[-1])
        h16_dpc = self.h16_dpc(ms_feat[-2])
        h8_dpc = self.h8_dpc(ms_feat[-3])
        h4_dpc = self.h4_dpc(ms_feat[-4])

        h32x16_int = self.h32x16(h32_dpc)
        h16x8_int = self.h16x8(h16_dpc)
        h8x4_int = self.h8x4(h8_dpc)
        # h4x2_int = self.h4x2(h4_dpc)
        # h2x1_int = self.h2x1(h4x2_int)

        h16_sum = self.elu(self.h16_sum_conv(torch.add(h32x16_int, h16_dpc)))
        h8_sum = self.elu(self.h8_sum_conv(torch.add(h16x8_int, h8_dpc)))
        h4_sum = self.elu(self.h4_sum_conv(torch.add(h8x4_int, h4_dpc)))

        h32x4 = self.h32x4(h32_dpc)
        h16x4 = self.h16x4(h16_sum)
        h8x4 = self.h8x4(h8_sum)
        h4x4 = h4_sum

        # Concat these 4 scales
        h4_cat = torch.cat([h32x4, h16x4, h8x4, h4x4], dim=1)
        h4_cat = self.elu(self.h4_cat_conv(h4_cat))
        h4_cat = self.make_out_ch(h4_cat)

        return h4_cat


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.ch_mapper = nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)

        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.BatchNorm2d(planes)

        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not stride == 1:
                self.norm3 = nn.Sequential()

        # if stride == 1:
        #     self.downsample = None
        #
        # else:
        #     self.downsample = nn.Sequential(nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        # Channel mapper. Does the job of downsample as well
        x = self.ch_mapper(x)

        # if self.downsample is not None:
        #     x = self.downsample(x)

        return self.relu(x + y)


class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, norm_fn='batch'):
        super(BasicBlock, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.norm_fn = norm_fn

        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(96, stride=1)
        self.layer3 = self._make_layer(128, stride=2)

        self.ch_mapper = nn.Conv2d(128, out_ch, kernel_size=1)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_ch, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_ch = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.ch_mapper(x)

        return x

class BasicMotionEncoder(nn.Module):
    def __init__(self, corr_levels, corr_radius):
        super(BasicMotionEncoder, self).__init__()
        cor_planes = corr_levels * (2 * corr_radius + 1)**2
        self.convc1 = nn.Conv2d(cor_planes, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv = nn.Conv2d(64+192, 128-2, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)

class BasicUpdateBlock(nn.Module):
    def __init__(self, corr_levels, corr_radius, input_dim=128, hidden_dim=128):
        super(BasicUpdateBlock, self).__init__()
        self.encoder = BasicMotionEncoder(corr_levels, corr_radius)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=input_dim + hidden_dim)
        self.flow_head = FlowHead(input_dim=hidden_dim, hidden_dim=256)

        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64 * 9, 1, padding=0))

    def forward(self, net, inp, corr, flow):
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)

        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)

        # scale mask to balance gradients
        mask = .25 * self.mask(net)
        return net, mask, delta_flow


class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))

        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))


    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q

        return h

def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img

class CorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # all pairs correlation
        corr = CorrBlock.corr(fmap1, fmap2)

        batch, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.reshape(batch * h1 * w1, dim, h2, w2)

        self.corr_pyramid.append(corr)
        for i in range(self.num_levels - 1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            dx = torch.linspace(-r, r, 2 * r + 1)
            dy = torch.linspace(-r, r, 2 * r + 1)
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1).to(coords.device)

            centroid_lvl = coords.reshape(batch * h1 * w1, 1, 1, 2) / 2 ** i
            delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht * wd)
        fmap2 = fmap2.view(batch, dim, ht * wd)

        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return corr / torch.sqrt(torch.tensor(dim).float())

def upflow8(flow, mode='bilinear'):
    new_size = (8 * flow.shape[2], 8 * flow.shape[3])
    return 8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)


class OpticalFlowHeadRAFT(nn.Module):
    def __init__(self, ms_in_channels, hidden_dim=128, context_dim=128, corr_levels=4, corr_radius=4, iters=12, **kwargs):
        super(OpticalFlowHeadRAFT, self).__init__()
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.corr_levels = corr_levels
        self.corr_radius = corr_radius
        self.iters = iters

        self.scale_merger = MultiScaleMergeRAFT(ms_in_channels, 64, 64)
        self.fnet = BasicBlock(in_ch=64, out_ch=256, norm_fn='instance')
        self.cnet = BasicBlock(in_ch=64, out_ch=self.hidden_dim + self.context_dim, norm_fn='batch')
        self.update_block = BasicUpdateBlock(self.corr_levels, self.corr_radius, input_dim=128, hidden_dim=hidden_dim)

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8 * H, 8 * W)

    def coords_grid(self, batch, ht, wd):
        coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
        coords = torch.stack(coords[::-1], dim=0).float()
        return coords[None].repeat(batch, 1, 1, 1)

    def initialise_flow(self, feat, flow_shape):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = flow_shape
        coords0 = self.coords_grid(N, H // 8, W // 8).to(feat.device)
        coords1 = self.coords_grid(N, H // 8, W // 8).to(feat.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def forward(self, ms_feat_prev, ms_feat_curr, upsample=True, flow_init=None, flow_shape=None):
        merged_feat_prev = self.scale_merger(ms_feat_prev)
        merged_feat_curr = self.scale_merger(ms_feat_curr)

        fmap_prev = self.fnet(merged_feat_prev)
        fmap_curr = self.fnet(merged_feat_curr)

        corr_fn = CorrBlock(fmap_prev, fmap_curr, radius=self.corr_radius)

        cmap_prev = self.cnet(merged_feat_prev)
        net, inp = torch.split(cmap_prev, [self.hidden_dim, self.context_dim], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)

        coords0, coords1 = self.initialise_flow(fmap_prev, flow_shape)

        if flow_init is not None:
            coords1 = flow_init + coords1

        flow_predictions = []
        for itr in range(self.iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)  # Index correlation volume

            flow = coords1 - coords0
            net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # Upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)

            flow_predictions.append(flow_up)

        flow_final = coords1 - coords0
        flow_final_up = flow_up

        return flow_predictions, flow_final, flow_final_up
