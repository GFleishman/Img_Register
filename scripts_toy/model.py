import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_block(in_ch, out_ch, kernel_sz=3, pad=1, bias=False):
    """
    Two 3D convolutional layer: 3D conv + batch norm + ReLu 
    Args:
        in_ch: number of input channels
        out_ch: number of output channels
        kernel_sz: size of the convolutional kernel
        pad: amount of zero padding added to the sides of input
        bias: if add a learnable bias to the output
    """
    return nn.Sequential(
        nn.Conv3d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_sz, padding=pad, bias=bias),
        nn.BatchNorm3d(num_features=out_ch),
        nn.LeakyReLU(inplace=True),
        nn.Conv3d(in_channels=out_ch, out_channels=out_ch, kernel_size=kernel_sz, padding=pad, bias=bias),
        nn.BatchNorm3d(num_features=out_ch),
        nn.LeakyReLU(inplace=True)
    )


class SimpleUnet(nn.Module):
    """
    3D UNet model as generator
    """
    def __init__(self, in_channels=1, base_filters=32, out_channels=3):
        super().__init__()
        self.conv_down1 = _conv_block(in_ch=in_channels, out_ch=base_filters)
        self.conv_down2 = _conv_block(in_ch=base_filters, out_ch=2*base_filters)
        self.conv_down3 = _conv_block(in_ch=2*base_filters, out_ch=4*base_filters)

        self.maxpool = nn.MaxPool3d(kernel_size=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        self.conv_up2 = _conv_block(in_ch=2*base_filters+4*base_filters, out_ch=2*base_filters)
        self.conv_up1 = _conv_block(in_ch=base_filters+2*base_filters, out_ch=base_filters)

        self.conv_out = nn.Conv3d(in_channels=base_filters, out_channels=out_channels, kernel_size=1)


    def forward(self, img):
        down1 = self.conv_down1(img) # 16x16x16
        x = self.maxpool(down1) # 8x8x8

        down2 = self.conv_down2(x) # 8x8x8
        x = self.maxpool(down2) # 4x4x4

        x = self.conv_down3(x) # 4x4x4
        
        x = self.upsample(x) # 8x8x8
        x = torch.cat([down2, x], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x) # 16x16x16
        x = torch.cat([down1, x], dim=1)
        x = self.conv_up1(x)

        phi = self.conv_out(x)
        return phi