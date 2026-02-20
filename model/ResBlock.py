import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, in_channels, inner_channel, out_channels):
        super(ResBlock, self).__init__()
        self.in_channels = in_channels
        self.inner_channel = inner_channel
        self.out_channels = out_channels

        self.mainstream = nn.Sequential(
            nn.Conv2d(self.in_channels, self.inner_channel, 1, 1, 0, bias=False),
            nn.GroupNorm(num_groups=16, num_channels=self.inner_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inner_channel, self.inner_channel, 3, 1, 1, bias=False),
            nn.GroupNorm(num_groups=16, num_channels=self.inner_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inner_channel, self.out_channels, 1, 1, 0, bias=False),
            nn.GroupNorm(num_groups=16, num_channels=self.out_channels),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0, bias=False),
            nn.GroupNorm(num_groups=16, num_channels=self.out_channels),
            )
        # self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.mainstream(x)
        x = self.shortcut(x)
        out = out + x
        # out = self.act(out)
        return out


#resbock with half instance normalization
class HinBlock(nn.Module):
    def __init__(self, in_size, out_size):
        super(HinBlock, self).__init__()
        # Identity shortcut
        self.identity = nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0)

        # Depthwise separable convolution for the first layer
        self.depthwise_conv_1 = nn.Conv2d(in_size, in_size, kernel_size=3, stride=1, padding=1, groups=in_size, bias=False)
        self.pointwise_conv_1 = nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=True)

        # Instance normalization on half the output channels
        self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)

        # Depthwise separable convolution for the second layer
        self.depthwise_conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, stride=1, padding=1, groups=out_size, bias=False)
        self.pointwise_conv_2 = nn.Conv2d(out_size, out_size, kernel_size=1, stride=1, padding=0, bias=True)

        # Activation layers
        self.relu_1 = nn.LeakyReLU(0.1, inplace=True)
        self.relu_2 = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        # First depthwise separable convolution
        out = self.depthwise_conv_1(x)
        out = self.pointwise_conv_1(out)

        # Split, normalize part of it, and concatenate
        out_1, out_2 = torch.chunk(out, 2, dim=1)
        out = torch.cat([self.norm(out_1), out_2], dim=1)

        # Apply activation
        out = self.relu_1(out)

        # Second depthwise separable convolution
        out = self.depthwise_conv_2(out)
        out = self.pointwise_conv_2(out)

        # Add the identity shortcut
        out += self.identity(x)


        return out