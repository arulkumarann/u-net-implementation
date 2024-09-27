import torch
import torch.nn as nn

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
        
        self.output = OutputBlock(64, 2)  # 2classes for binary segentation
    
    def forward(self, x):
        # Encoder path
        x1_pooled, x1 = self.enc1(x)
        x2_pooled, x2 = self.enc2(x1_pooled)
        x3_pooled, x3 = self.enc3(x2_pooled)
        x4_pooled, x4 = self.enc4(x3_pooled)
        
        #bottleneck
        x_bottleneck = self.bottleneck(x4_pooled)
        
        #decoder path
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
