import torch.nn as nn
import torch

# courtesy: https://github.com/darkstar112358/fast-neural-style/blob/master/neural_style/transformer_net.py
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class Encoder(nn.Module):
    def __init__(self, input_nc, output_nc, ngf):
        super(Encoder,self).__init__()
        
        self.layer1 = nn.Sequential(nn.ReflectionPad2d((3,3,3,3)),
                                    nn.Conv2d(input_nc,ngf,kernel_size=7,stride=1),
                                    nn.InstanceNorm2d(ngf),
                                    nn.ReLU(True))
        
        self.layer2 = nn.Sequential(nn.Conv2d(ngf,ngf*2,kernel_size=3,stride=2,padding=1),
                                   nn.InstanceNorm2d(ngf*2),
                                   nn.ReLU(True))
        
        self.layer3 = nn.Sequential(nn.Conv2d(ngf*2,ngf*4,kernel_size=3,stride=2,padding=1),
                                   nn.InstanceNorm2d(ngf*4),
                                   nn.ReLU(True))

        self.layer4 = nn.Sequential(ResidualBlock(ngf*4,ngf*4),
                                    ResidualBlock(ngf*4,ngf*4),
                                    ResidualBlock(ngf*4,ngf*4))
        
    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out
