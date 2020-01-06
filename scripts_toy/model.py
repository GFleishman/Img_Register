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


# original u-net paper uses 2x2 kernels here, but asymmetrical padding is not supported, so trying with 3x3x3
# TODO: docstring
def _up_conv_block(in_ch, out_ch, kernel_sz=3, pad=1, bias=False):

    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
        nn.Conv3d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_sz, padding=pad, bias=bias),
        nn.BatchNorm3d(num_features=out_ch)
    )


class SimpleUnet(nn.Module):
    """
    Traditional UNet model as generator
    """
    def __init__(self, in_channels=1, base_filters=32, out_channels=3):
        super().__init__()
        self.conv_down1 = _conv_block(in_ch=in_channels, out_ch=base_filters)
        self.conv_down2 = _conv_block(in_ch=base_filters, out_ch=2*base_filters)
        self.conv_down3 = _conv_block(in_ch=2*base_filters, out_ch=4*base_filters)

        self.maxpool = nn.MaxPool3d(kernel_size=2)

        self.conv_up2p5 = _up_conv_block(in_ch=4*base_filters, out_ch=2*base_filters)
        self.conv_up2 = _conv_block(in_ch=4*base_filters, out_ch=2*base_filters)

        self.conv_up1p5 = _up_conv_block(in_ch=2*base_filters, out_ch=base_filters)
        self.conv_up1 = _conv_block(in_ch=2*base_filters, out_ch=base_filters)

        self.conv_out = nn.Conv3d(in_channels=base_filters, out_channels=out_channels, kernel_size=1, bias=False)
        self.conv_out.weight.data.fill_(0)


    def forward(self, img):
        down1 = self.conv_down1(img) # 64x64x64
        x = self.maxpool(down1) # 32x32x32

        down2 = self.conv_down2(x)
        x = self.maxpool(down2) # 16x16x16

        x = self.conv_down3(x) # 16x16x16
       
        x = self.conv_up2p5(x) 
        x = torch.cat([down2, x], dim=1)
        x = self.conv_up2(x)

        x = self.conv_up1p5(x)
        x = torch.cat([down1, x], dim=1)
        x = self.conv_up1(x)

        phi = self.conv_out(x)
        return phi



