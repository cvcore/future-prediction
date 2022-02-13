import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaleFeatureMap(nn.Module):
    def __init__(self, in_channels, out_channels, scale_ratio=2., mode="nearest"):
        super(ScaleFeatureMap, self).__init__()
        self.elu = nn.ELU()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=2*out_channels, bias=False, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=2*out_channels, out_channels=out_channels, bias=False, kernel_size=3, stride=1, padding=1)
        self.scale_ratio = scale_ratio
        self.mode = mode

    def forward(self, x):
        scaled_x = F.interpolate(x, scale_factor=self.scale_ratio, mode=self.mode)
        conv1 = self.elu(self.conv1(scaled_x))
        out = self.elu(self.conv2(conv1))
        return out


class Reduction1x1(nn.Sequential):
    def __init__(self, num_in_channels, num_out_channels, max_depth, is_final=False):
        super(Reduction1x1, self).__init__()
        self.max_depth = max_depth
        self.is_final = is_final
        self.sigmoid = nn.Sigmoid()
        self.reduction = torch.nn.Sequential()

        while num_out_channels >= 4:
            if num_out_channels < 8:
                if self.is_final:
                    self.reduction.add_module("final", torch.nn.Sequential(nn.Conv2d(num_in_channels, out_channels=1, bias=False, kernel_size=1, stride=1, padding=0), nn.Sigmoid()))
                else:
                    self.reduction.add_module("plane_params", torch.nn.Conv2d(num_in_channels, out_channels=3, bias=False, kernel_size=1, stride=1, padding=0))
                break
            else:
                self.reduction.add_module("inter_{}_{}".format(num_in_channels, num_out_channels),
                                          nn.Sequential(nn.Conv2d(in_channels=num_in_channels, out_channels=num_out_channels, bias=False, kernel_size=1, stride=1, padding=0), nn.ELU()))

            num_in_channels = num_out_channels
            num_out_channels = num_out_channels // 2

    def forward(self, net):
        net = self.reduction.forward(net)
        if not self.is_final:
            theta = self.sigmoid(net[:, 0, :, :]) * math.pi / 3
            phi = self.sigmoid(net[:, 1, :, :]) * math.pi * 2
            dist = self.sigmoid(net[:, 2, :, :]) * self.max_depth
            n1 = torch.mul(torch.sin(theta), torch.cos(phi)).unsqueeze(1)
            n2 = torch.mul(torch.sin(theta), torch.sin(phi)).unsqueeze(1)
            n3 = torch.cos(theta).unsqueeze(1)
            n4 = dist.unsqueeze(1)
            net = torch.cat([n1, n2, n3, n4], dim=1)

        return net


class LocalPlanarGuidance(nn.Module):
    def __init__(self, upratio):
        super(LocalPlanarGuidance, self).__init__()
        self.upratio = upratio
        self.u = torch.arange(self.upratio).reshape([1, 1, self.upratio]).float()
        self.v = torch.arange(int(self.upratio)).reshape([1, self.upratio, 1]).float()
        self.upratio = float(upratio)

    def forward(self, plane_eq):
        plane_eq_expanded = torch.repeat_interleave(plane_eq, int(self.upratio), 2)
        plane_eq_expanded = torch.repeat_interleave(plane_eq_expanded, int(self.upratio), 3)
        n1 = plane_eq_expanded[:, 0, :, :]
        n2 = plane_eq_expanded[:, 1, :, :]
        n3 = plane_eq_expanded[:, 2, :, :]
        n4 = plane_eq_expanded[:, 3, :, :]

        u = self.u.repeat(plane_eq.size(0), plane_eq.size(2) * int(self.upratio), plane_eq.size(3)).cuda()
        u = (u - (self.upratio - 1) * 0.5) / self.upratio

        v = self.v.repeat(plane_eq.size(0), plane_eq.size(2), plane_eq.size(3) * int(self.upratio)).cuda()
        v = (v - (self.upratio - 1) * 0.5) / self.upratio

        return n4 / (n1*u + n2*v + n3)


class AtrousConvolution(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation, apply_bn_first=True):
        super(AtrousConvolution, self).__init__()
        self.atrous_conv = torch.nn.Sequential()
        if apply_bn_first:
            self.atrous_conv.add_module("first_bn", nn.BatchNorm2d(in_channels, momentum=0.01, affine=True, track_running_stats=True, eps=1.1e-5))
        self.atrous_conv.add_module("aconv_sequence", nn.Sequential(nn.ReLU(),
                                                                    nn.Conv2d(in_channels=in_channels, out_channels=2*out_channels, bias=False, kernel_size=1, stride=1, padding=0),
                                                                    nn.BatchNorm2d(num_features=2*out_channels, momentum=0.01, affine=True, track_running_stats=True),
                                                                    nn.ReLU(),
                                                                    nn.Conv2d(in_channels=2*out_channels, out_channels=out_channels, bias=False, kernel_size=3, stride=1, padding=(dilation, dilation), dilation=dilation)))

    def forward(self, x):
        return self.atrous_conv.forward(x)


class FlowDPCSmall(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FlowDPCSmall, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=2 * out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=2 * out_channels, out_channels=2 * out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(in_channels=2 * out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn3 = nn.BatchNorm2d(num_features=out_channels)
        self.elu = nn.ELU(inplace=True)

    def forward(self, x):
        x = self.elu(self.conv1(x))
        x = self.elu(self.conv2(x))
        x = self.elu(self.bn3(self.conv3(x)))
        return x


class SemanticDepthHead(nn.Module):
    def __init__(self, ms_in_channels, sem_feat_in_channels=512, out_channels=128, max_depth=10, class_count=8):
        super(SemanticDepthHead, self).__init__()

        self.elu = nn.ELU(inplace=True)
        self.max_depth = max_depth

        # Merge all scales into one H/4 scale
        self.h32_dpc = FlowDPCSmall(ms_in_channels[-1], out_channels)
        self.h32x16 = ScaleFeatureMap(out_channels, out_channels, scale_ratio=2)
        self.h16_dpc = FlowDPCSmall(ms_in_channels[-2], out_channels)
        self.h16x8 = ScaleFeatureMap(out_channels, out_channels, scale_ratio=2)
        self.h8_dpc = FlowDPCSmall(ms_in_channels[-3], out_channels)
        self.h8x4 = ScaleFeatureMap(out_channels, out_channels, scale_ratio=2)
        self.h4_dpc = FlowDPCSmall(ms_in_channels[-4], out_channels)

        self.h32x4 = ScaleFeatureMap(out_channels, out_channels, scale_ratio=8)
        self.h16x4 = ScaleFeatureMap(out_channels, out_channels, scale_ratio=4)
        self.h8x4 = ScaleFeatureMap(out_channels, out_channels, scale_ratio=2)
        self.h4x2 = ScaleFeatureMap(out_channels, out_channels, scale_ratio=2)
        self.h2x1 = ScaleFeatureMap(out_channels, out_channels, scale_ratio=2)

        self.h16_sum_conv = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.h8_sum_conv = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.h4_sum_conv = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)

        self.h4_cat_conv = nn.Conv2d(4 * out_channels, out_channels, 3, 1, 1, bias=False)

        # ASPP
        self.daspp4_3 = AtrousConvolution(out_channels, out_channels, 3, apply_bn_first=False)
        self.daspp4_6 = AtrousConvolution(2 * out_channels, out_channels, 6)
        self.daspp4_12 = AtrousConvolution(3 * out_channels, out_channels, 12)
        self.daspp4_18 = AtrousConvolution(4 * out_channels, out_channels, 18)
        self.daspp4_24 = AtrousConvolution(5 * out_channels, out_channels, 24)
        self.daspp4_conv = torch.nn.Sequential(nn.Conv2d(6 * out_channels, out_channels, 3, 1, 1, bias=False), nn.ELU())

        # self.sem_feat_bottleneck = torch.nn.Sequential(nn.Conv2d(sem_feat_in_channels, out_channels // 4, 1, 1, 0, bias=False),
        #                                              nn.ELU(),
        #                                              nn.Conv2d(out_channels // 4, out_channels // 4, 3, 1, 1, bias=False),
        #                                              nn.ELU(),
        #                                              nn.Conv2d(out_channels // 4, out_channels, 1, 1, 0, bias=False),
        #                                              nn.ELU())

        self.sem_daspp_merge = torch.nn.Sequential(nn.Conv2d(out_channels + sem_feat_in_channels, 2 * out_channels, 3, 1, 1, bias=False),
                                                   nn.ELU(),
                                                   nn.Conv2d(2 * out_channels, 2 * out_channels, 3, 1, 1, bias=False),
                                                   nn.ELU(),
                                                   nn.Conv2d(2 * out_channels, out_channels, 3, 1, 1, bias=False),
                                                   nn.ELU())
        # self.sem_daspp_chmap4x4 = torch.nn.Sequential(nn.Conv2d(out_channels + sem_feat_in_channels, out_channels, 1, 1, 0, bias=False), nn.ELU())

        # The BTS stuff
        self.reduce4x4 = Reduction1x1(out_channels, out_channels, max_depth)
        self.lpg4x4 = LocalPlanarGuidance(4)

        self.upconv4x2 = ScaleFeatureMap(out_channels, out_channels, mode='nearest')
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=0.01, affine=True, eps=1.1e-5)
        self.conv2 = torch.nn.Sequential(nn.Conv2d(2 * out_channels + 1, out_channels, 3, 1, 1, bias=False), nn.ELU())

        self.reduce2x2 = Reduction1x1(out_channels, out_channels, max_depth)
        self.lpg2x2 = LocalPlanarGuidance(2)

        self.upconv2x1 = ScaleFeatureMap(out_channels, out_channels, mode='nearest')
        self.reduce1x1 = Reduction1x1(out_channels, out_channels, max_depth, is_final=True)
        self.conv1 = torch.nn.Sequential(nn.Conv2d(2 * out_channels + 3, out_channels, 3, 1, 1, bias=False), nn.ELU())

        # Incorporate the class-wise depth estimate
        self.class_conv1 = nn.Conv2d(out_channels, class_count, 3, 1, 1, bias=False)
        self.class_conv2 = nn.Conv2d(class_count, class_count, 3, 1, 1, bias=False)
        self.class_conv3 = nn.Conv2d(class_count, class_count, 3, 1, 1, bias=False)
        self.class_get_depth = nn.Conv2d(class_count, 1, 3, 1, 1, bias=False)

        self.summed_conv1 = nn.Conv2d(1, 1, 3, 1, 1, bias=False)
        self.summed_conv2 = nn.Conv2d(1, 1, 3, 1, 1, bias=False)
        # self.summed_conv3 = nn.Conv2d(1, 1, 3, 1, 1, bias=False)

        self.class_sigmoid = nn.Sigmoid()
        self.class_elu = nn.ELU()

        # After conv1, we get the feature map of size (B, 128, H, H)
        self.get_depth = torch.nn.Sequential(nn.Conv2d(out_channels, 1, 3, 1, 1, bias=False), nn.Sigmoid())

    def forward(self, ms_feat, sem_feat):
        # The normal depth head
        # print(ms_feat[-1].shape, ms_feat[-2].shape, ms_feat[-3].shape, ms_feat[-4].shape)

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

        # Concatenate the semantic features here!
        sem_h4_cat = torch.cat([h4_cat, sem_feat], dim=1)
        sem_h4_merge = self.sem_daspp_merge(sem_h4_cat)

        # ASPP on the concatenated layer
        daspp4_3 = self.daspp4_3(sem_h4_merge)
        concat4_1 = torch.cat([sem_h4_merge, daspp4_3], dim=1)
        daspp4_6 = self.daspp4_6(concat4_1)
        concat4_2 = torch.cat([concat4_1, daspp4_6], dim=1)
        daspp4_12 = self.daspp4_12(concat4_2)
        concat4_3 = torch.cat([concat4_2, daspp4_12], dim=1)
        daspp4_18 = self.daspp4_18(concat4_3)
        concat4_4 = torch.cat([concat4_3, daspp4_18], dim=1)
        daspp4_24 = self.daspp4_24(concat4_4)
        concat4_daspp = torch.cat([sem_h4_merge, daspp4_3, daspp4_6, daspp4_12, daspp4_18, daspp4_24], dim=1)
        daspp_4_feat = self.daspp4_conv(concat4_daspp)

        # Process the semantic features
        # sem_feat_bottleneck = self.sem_feat_bottleneck(sem_feat)

        # # Concatenate the semantic features here!
        # sem_daspp4_cat = torch.cat([daspp_4_feat, sem_feat], dim=1)
        # sem_daspp4_merge = self.sem_daspp_merge(sem_daspp4_cat)

        # The multi-scale depth stuff
        # Scale H/4
        reduce4x4 = self.reduce4x4(daspp_4_feat)
        plane_normal_4x4 = reduce4x4[:, :3, :, :]
        plane_normal_4x4 = F.normalize(plane_normal_4x4, 2, 1)
        plane_dist_4x4 = reduce4x4[:, 3, :, :]
        plane_eq_4x4 = torch.cat([plane_normal_4x4, plane_dist_4x4.unsqueeze(1)], 1)
        depth_4x4 = self.lpg4x4(plane_eq_4x4)
        depth_4x4_scaled = depth_4x4.unsqueeze(1) / self.max_depth
        depth_4x4_scaled_ds = F.interpolate(depth_4x4_scaled, scale_factor=0.5, mode='nearest')

        upconv4x2 = self.bn2(self.upconv4x2(daspp_4_feat))
        concat2 = torch.cat([upconv4x2, h4x2_int, depth_4x4_scaled_ds], dim=1)
        out2 = self.conv2(concat2)

        # Scale H/2
        reduce2x2 = self.reduce2x2(out2)
        plane_normal_2x2 = reduce2x2[:, :3, :, :]
        plane_normal_2x2 = F.normalize(plane_normal_2x2, 2, 1)
        plane_dist_2x2 = reduce2x2[:, 3, :, :]
        plane_eq_2x2 = torch.cat([plane_normal_2x2, plane_dist_2x2.unsqueeze(1)], 1)
        depth_2x2 = self.lpg2x2(plane_eq_2x2)
        depth_2x2_scaled = depth_2x2.unsqueeze(1) / self.max_depth
        depth_2x2_scaled_ds = F.interpolate(depth_2x2_scaled, scale_factor=1, mode='nearest')

        upconv2x1 = self.upconv2x1(out2)
        reduce1x1 = self.reduce1x1(upconv2x1)
        concat1 = torch.cat([upconv2x1, h2x1_int, reduce1x1, depth_4x4_scaled, depth_2x2_scaled], dim=1)

        # out1 = self.conv1(concat1)
        # final_depth = self.max_depth * self.get_depth(out1)
        #
        # return None, final_depth

        # Class-wise depth
        out1 = self.conv1(concat1)
        class_conv1 = self.class_conv1(out1)  # Old line
        class_conv2 = self.class_elu(self.class_conv2(class_conv1))
        class_conv3 = self.class_elu(self.class_conv3(class_conv2))
        summed_conv3 = self.class_get_depth(class_conv3)
        # summed_conv3 = torch.sum(class_conv1, dim=1).unsqueeze(1)

        class_depth = self.class_sigmoid(class_conv1) * self.max_depth
        final_depth = self.class_sigmoid(summed_conv3) * self.max_depth

        # out1 = self.conv1(concat1)
        # class_conv1 = self.class_conv1(out1)
        # summed_class = torch.sum(class_conv1, dim=1).unsqueeze(1)
        #
        # # Some smoothing and integration layers
        # summed_class = self.class_elu(self.summed_conv1(summed_class))
        # summed_class = self.class_elu(self.summed_conv2(summed_class))
        # summed_class = self.summed_conv3(summed_class)
        #
        # class_depth = self.class_sigmoid(class_conv1) * self.max_depth
        # final_depth = self.class_sigmoid(summed_class) * self.max_depth

        return out1, class_depth, final_depth

# class SemanticDepthHead(nn.Module):
#     def __init__(self, in_channels, out_channels=128, max_depth=100., class_count=8):
#         super(SemanticDepthHead, self).__init__()
#
#         self.elu = nn.ELU(inplace=True)
#         self.max_depth = max_depth
#
#         # Convolve the input semantic features to H/8 and reduce the channels by a factor of 2
#         # Get the semantic features ready for use with BTS
#         self.sem_ds4x8 = ScaleFeatureMap(in_channels, in_channels // 2, scale_ratio=0.5, mode="bilinear")
#         self.sem_chmap8x8 = torch.nn.Sequential(nn.Conv2d(in_channels // 2, out_channels, 1, 1, 0, bias=False), nn.ELU())
#
#         self.sem_conv4x4 = ScaleFeatureMap(in_channels, in_channels // 2, scale_ratio=1, mode="bilinear")
#         self.sem_chmap4x4 = torch.nn.Sequential(nn.Conv2d(in_channels // 2, out_channels, 1, 1, 0, bias=False), nn.ELU())
#
#         self.sem_upconv4x2 = ScaleFeatureMap(in_channels, in_channels // 2, mode="bilinear")
#         self.sem_chmap2x2 = torch.nn.Sequential(nn.Conv2d(in_channels // 2, out_channels, 1, 1, 0, bias=False), nn.ELU())
#
#         self.sem_upconv4x1 = ScaleFeatureMap(in_channels, in_channels // 2, scale_ratio=4, mode="bilinear")
#         self.sem_chmap1x1 = torch.nn.Sequential(nn.Conv2d(in_channels // 2, out_channels, 1, 1, 0, bias=False), nn.ELU())
#
#         # The BTS stuff
#         self.reduce8x8 = Reduction1x1(out_channels, out_channels, max_depth)
#         self.lpg8x8 = LocalPlanarGuidance(8)
#
#         self.upconv8x4 = ScaleFeatureMap(out_channels, out_channels, mode="bilinear")
#         self.bn4 = nn.BatchNorm2d(out_channels, momentum=0.01, affine=True, eps=1.1e-5)
#         self.conv4 = torch.nn.Sequential(nn.Conv2d(out_channels + 1, out_channels, 3, 1, 1, bias=False), nn.ELU())
#
#         self.reduce4x4 = Reduction1x1(out_channels, out_channels, max_depth)
#         self.lpg4x4 = LocalPlanarGuidance(4)
#
#         self.upconv4x2 = ScaleFeatureMap(out_channels, out_channels, mode='bilinear')
#         self.bn2 = nn.BatchNorm2d(out_channels, momentum=0.01, affine=True, eps=1.1e-5)
#         self.conv2 = torch.nn.Sequential(nn.Conv2d(2 * out_channels + 1, out_channels, 3, 1, 1, bias=False), nn.ELU())
#
#         self.reduce2x2 = Reduction1x1(out_channels, out_channels, max_depth)
#         self.lpg2x2 = LocalPlanarGuidance(2)
#
#         self.upconv2x1 = ScaleFeatureMap(out_channels, out_channels, mode='bilinear')
#         self.reduce1x1 = Reduction1x1(out_channels, out_channels, max_depth, is_final=True)
#         self.conv1 = torch.nn.Sequential(nn.Conv2d(2 * out_channels + 4, out_channels, 3, 1, 1, bias=False), nn.ELU())
#
#         # Incorporate the class-wise depth estimate
#         self.class_conv1 = nn.Conv2d(out_channels, class_count, 3, 1, 1, bias=False)
#         self.class_conv2 = nn.Conv2d(class_count, class_count, 3, 1, 1, bias=False)
#         self.class_conv3 = nn.Conv2d(class_count, class_count, 3, 1, 1, bias=False)
#         self.class_get_depth = nn.Conv2d(class_count, 1, 3, 1, 1, bias=False)
#
#         self.class_sigmoid = nn.Sigmoid()
#
#         # After conv1, we get the feature map of size (B, 128, H, H)
#         self.get_depth = torch.nn.Sequential(nn.Conv2d(out_channels, 1, 3, 1, 1, bias=False), nn.Sigmoid())
#
#     def forward(self, sem_feat):
#         # Downsize the semantic features
#         sem_feat4x8 = self.sem_ds4x8(sem_feat)
#         sem_feat4x8 = self.sem_chmap8x8(sem_feat4x8)
#
#         # Scale H/8
#         reduce8x8 = self.reduce8x8(sem_feat4x8)
#         plane_normal8x8 = reduce8x8[:, :3, :, :]
#         plane_normal8x8 = F.normalize(plane_normal8x8, 2, 1)
#         plane_dist8x8 = reduce8x8[:, 3, :, :]
#         plane_eq8x8 = torch.cat([plane_normal8x8, plane_dist8x8.unsqueeze(1)], 1)
#         depth8x8 = self.lpg8x8(plane_eq8x8)
#         depth8x8_scaled = depth8x8.unsqueeze(1) / self.max_depth
#         depth8x8_scaled_ds = F.interpolate(depth8x8_scaled, scale_factor=0.25, mode="bilinear")
#
#         sem_feat4x4 = self.sem_chmap4x4(self.sem_conv4x4(sem_feat))
#         upconv8x4 = self.bn4(sem_feat4x4)
#         concat4 = torch.cat([upconv8x4, depth8x8_scaled_ds], dim=1)
#         out4 = self.conv4(concat4)
#
#         # Scale H/4
#         reduce4x4 = self.reduce4x4(out4)
#         plane_normal4x4 = reduce4x4[:, :3, :, :]
#         plane_normal4x4 = F.normalize(plane_normal4x4, 2, 1)
#         plane_dist4x4 = reduce4x4[:, 3, :, :]
#         plane_eq4x4 = torch.cat([plane_normal4x4, plane_dist4x4.unsqueeze(1)], 1)
#         depth4x4 = self.lpg4x4(plane_eq4x4)
#         depth4x4_scaled = depth4x4.unsqueeze(1) / self.max_depth
#         depth4x4_scaled_ds = F.interpolate(depth4x4_scaled, scale_factor=0.5, mode='bilinear')
#
#         sem_feat4x2 = self.sem_chmap2x2(self.sem_upconv4x2(sem_feat))
#         upconv4x2 = self.bn2(self.upconv4x2(out4))
#         concat2 = torch.cat([upconv4x2, sem_feat4x2, depth4x4_scaled_ds], dim=1)
#         out2 = self.conv2(concat2)
#
#         # Scale H/2
#         reduce2x2 = self.reduce2x2(out2)
#         plane_normal2x2 = reduce2x2[:, :3, :, :]
#         plane_normal2x2 = F.normalize(plane_normal2x2, 2, 1)
#         plane_dist2x2 = reduce2x2[:, 3, :, :]
#         plane_eq2x2 = torch.cat([plane_normal2x2, plane_dist2x2.unsqueeze(1)], 1)
#         depth2x2 = self.lpg2x2(plane_eq2x2)
#         depth2x2_scaled = depth2x2.unsqueeze(1) / self.max_depth
#         depth2x2_scaled_ds = F.interpolate(depth2x2_scaled, scale_factor=1, mode='bilinear')
#
#         sem_feat4x1 = self.sem_chmap1x1(self.sem_upconv4x1(sem_feat))
#         upconv2x1 = self.upconv2x1(out2)
#         reduce1x1 = self.reduce1x1(upconv2x1)
#         concat1 = torch.cat([upconv2x1, sem_feat4x1, reduce1x1, depth8x8_scaled, depth4x4_scaled, depth2x2_scaled], dim=1)
#         # out1 = self.conv1(concat1)
#         # final_depth = self.max_depth * self.get_depth(out1)
#
#         # Class-wise depth
#         out1 = self.conv1(concat1)
#         class_conv1 = self.class_conv1(out1)
#         class_conv2 = self.elu(self.class_conv2(class_conv1))
#         class_conv3 = self.elu(self.class_conv3(class_conv2))
#         summed_conv3 = torch.sum(class_conv3, dim=1).unsqueeze(1)
#
#         class_depth = self.class_sigmoid(class_conv1) * self.max_depth
#         final_depth = self.class_sigmoid(summed_conv3) * self.max_depth
#
#         return out1, class_depth, final_depth
#         # return out1, final_depth
