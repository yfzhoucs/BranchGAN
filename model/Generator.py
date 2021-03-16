import torch.nn as nn
import torch


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


class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf):
        super(Generator,self).__init__()

        self.layer1 = nn.Sequential(ResidualBlock(ngf*4,ngf*4),
                                    ResidualBlock(ngf*4,ngf*4),
                                    ResidualBlock(ngf*4,ngf*4))
        
        self.layer2 = nn.Sequential(nn.ConvTranspose2d(ngf*4, ngf*2, 3, 2, 1, 1),
                                     nn.InstanceNorm2d(ngf*2),
                                     nn.ReLU(True))
        
        self.layer3 = nn.Sequential(nn.ConvTranspose2d(ngf*2, ngf, 3, 2, 1, 1),
                                    nn.InstanceNorm2d(ngf),
                                    nn.ReLU(True))
        
        self.layer4 = nn.Sequential(nn.ReflectionPad2d((3,3,3,3)),
                                     nn.Conv2d(ngf,output_nc,kernel_size=7,stride=1),
                                     nn.Tanh())
        
    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out