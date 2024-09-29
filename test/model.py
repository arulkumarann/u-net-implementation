import torch
import torch.nn as nn


class ContractingBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ContractingBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x_pooled = self.pool(x)
        return x_pooled, x


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        return x


class ExpandingBlock(nn.Module):
    def __init__(self, in_features, out_channels):
        super(ExpandingBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_features, out_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x, skip_x):
        x = self.upconv(x)
        x = torch.cat((x, skip_x), dim=1)  #concatenate along the channel dimension
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        return x


class OutputBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(OutputBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.enc1 = ContractingBlock(1, 64)
        self.enc2 = ContractingBlock(64, 128)
        self.enc3 = ContractingBlock(128, 256)
        self.enc4 = ContractingBlock(256, 512)

        self.bottleneck = Bottleneck(512, 1024)

        self.dec4 = ExpandingBlock(1024, 512)
        self.dec3 = ExpandingBlock(512, 256)
        self.dec2 = ExpandingBlock(256, 128)
        self.dec1 = ExpandingBlock(128, 64)

        self.output = OutputBlock(64, 2)  #2classes for binary segmentation

    def forward(self, x):
        x1_pooled, x1 = self.enc1(x)
        x2_pooled, x2 = self.enc2(x1_pooled)
        x3_pooled, x3 = self.enc3(x2_pooled)
        x4_pooled, x4 = self.enc4(x3_pooled)

        x_bottleneck = self.bottleneck(x4_pooled)

        x = self.dec4(x_bottleneck, x4)
        x = self.dec3(x, x3)
        x = self.dec2(x, x2)
        x = self.dec1(x, x1)

        # Output
        output = self.output(x)
        return output


unet = UNet()
input_tensor = torch.randn(1, 1, 256, 256)  #batch size 1, 256x256 image
output = unet(input_tensor)
print(output.shape)  #should print torch.Size([1, 2, 256, 256])
