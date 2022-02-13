from torchvision import ops
import torch.nn as nn

class DeformConvPack(ops.DeformConv2d):

    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding,
                 dilation=1, groups=1, bias=True):

        super().__init__(in_channels, out_channels,
                         kernel_size, stride, padding, dilation, groups, bias)

        self.conv_offset_mask = nn.Conv2d(self.in_channels,
                                          self.groups * 2 * self.kernel_size[0] * self.kernel_size[1],
                                          kernel_size=self.kernel_size,
                                          stride=self.stride,
                                          padding=self.padding,
                                          bias=True)
        self.init_offset()

    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, input):
        offset = self.conv_offset_mask(input)
        orig_dtype = input.dtype
        input = input.float()
        offset = offset.float()
        out = super().forward(input, offset)
        out = out.to(orig_dtype)
        return out
