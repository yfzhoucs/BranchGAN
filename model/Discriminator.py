import torch.nn as nn
import torch


class Discriminator(nn.Module):
    def __init__(self,input_nc,ndf):
        super(Discriminator,self).__init__()
        
        self.layer1 = nn.Sequential(nn.Conv2d(input_nc,ndf,kernel_size=4,stride=2,padding=1),
                                 nn.LeakyReLU(0.2,inplace=True))
        
        self.layer2 = nn.Sequential(nn.Conv2d(ndf,ndf*2,kernel_size=4,stride=2,padding=1),
                                 nn.InstanceNorm2d(ndf*2),
                                 nn.LeakyReLU(0.2,inplace=True))
        
        self.layer3 = nn.Sequential(nn.Conv2d(ndf*2,ndf*4,kernel_size=4,stride=1,padding=1),
                                 nn.InstanceNorm2d(ndf*4),
                                 nn.LeakyReLU(0.2,inplace=True))
        
        self.layer4 = nn.Sequential(nn.Conv2d(ndf*4,1,kernel_size=4,stride=1,padding=1),
                                 nn.Sigmoid())
        

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out