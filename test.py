from __future__ import print_function
import argparse
import os
from os import listdir
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

from utils.dataset import DATASET
from utils.ImagePool import ImagePool
from model.Discriminator import Discriminator
from model.Generator import Generator
from model.Encoder import Encoder

parser = argparse.ArgumentParser(description='train pix2pix model')
parser.add_argument('--batchSize', type=int, default=1, help='with batchSize=1 equivalent to instance normalization.')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--outf', default='output_images/', help='folder to output images')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--dataPath', default='./datasets/cityscapes/val', help='path to training images')
parser.add_argument('--loadSize', type=int, default=128, help='scale image to this size')
parser.add_argument('--fineSize', type=int, default=128, help='random crop image to this size')
parser.add_argument('--poolSize', type=int, default=100, help='size of buffer in lsGAN, poolSize=0 indicates not using history')
parser.add_argument('--flip', type=int, default=0, help='1 for flipping image randomly, 0 for not')
parser.add_argument('--input_nc', type=int, default=3, help='channel number of input image')
parser.add_argument('--output_nc', type=int, default=3, help='channel number of output image')
parser.add_argument('--G_A', default='', help='path to pre-trained G_A')
parser.add_argument('--G_B', default='', help='path to pre-trained G_B')
parser.add_argument('--F', default='', help='path to pre-trained F')
parser.add_argument('--imgNum', type=int, default=500, help='image number')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True
##########   DATASET   ###########
datasetA = DATASET(os.path.join(opt.dataPath,'A'),opt.loadSize,opt.fineSize,opt.flip)
datasetB = DATASET(os.path.join(opt.dataPath,'B'),opt.loadSize,opt.fineSize,opt.flip)
loader_A = torch.utils.data.DataLoader(dataset=datasetA,
                                       batch_size=opt.batchSize,
                                       shuffle=False,
                                       num_workers=2)
loaderA = iter(loader_A)
loader_B = torch.utils.data.DataLoader(dataset=datasetB,
                                       batch_size=opt.batchSize,
                                       shuffle=False,
                                       num_workers=2)
loaderB = iter(loader_B)
ABPool = ImagePool(opt.poolSize)
BAPool = ImagePool(opt.poolSize)
loaderB = iter(loader_B)
###########   MODEL   ###########
# custom weights initialization called on netG and netD
ndf = opt.ndf
ngf = opt.ngf
nc = 3

G_A = Generator(opt.input_nc, opt.output_nc, opt.ngf)
G_B = Generator(opt.output_nc, opt.input_nc, opt.ngf)
F = Encoder(opt.input_nc, opt.output_nc, opt.ngf)

if(opt.G_A != ''):
    print('Warning! Loading pre-trained weights.')
    G_A.load_state_dict(torch.load(opt.G_A))
    G_B.load_state_dict(torch.load(opt.G_B))
    F.load_state_dict(torch.load(opt.F))
else:
    print('ERROR! G_AB and G_BA must be provided!')

if(opt.cuda):
    G_A.cuda()
    G_B.cuda()
    F.cuda()

###########   GLOBAL VARIABLES   ###########
input_nc = opt.input_nc
output_nc = opt.output_nc
fineSize = opt.fineSize

real_A = torch.FloatTensor(opt.batchSize, input_nc, fineSize, fineSize)
real_B = torch.FloatTensor(opt.batchSize, input_nc, fineSize, fineSize)
AB = torch.FloatTensor(opt.batchSize, output_nc, fineSize, fineSize)
BA = torch.FloatTensor(opt.batchSize, input_nc, fineSize, fineSize)
print(fineSize)


real_A = Variable(real_A)
real_B = Variable(real_B)
AB = Variable(AB)
BA = Variable(BA)

if(opt.cuda):
    real_A = real_A.cuda()
    real_B = real_B.cuda()
    AB = AB.cuda()
    BA = BA.cuda()

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def replace(oldstr,newstr,str):
    return str.replace(oldstr,newstr)

def find(str):
  if str.find('frankfurt') != -1:
    return 'frankfurt'
  elif str.find('lindau') != -1:
    return 'lindau'
  elif str.find('munster') != -1:
    return 'munster'


###########   Testing    ###########
def test():
    listA = [x for x in listdir(os.path.join(opt.dataPath,'A')) if is_image_file(x)]
    listB = [x for x in listdir(os.path.join(opt.dataPath,'B')) if is_image_file(x)]

    loaderA, loaderB = iter(loader_A), iter(loader_B)
    for i in range(0,opt.imgNum,opt.batchSize):
        print(i)

        imgA = loaderA.next()
        imgB = loaderB.next()

        real_A.data.resize_(imgA.size()).copy_(imgA)
        real_B.data.resize_(imgB.size()).copy_(imgB)

        AA = G_A(F(real_A))
        AB = G_B(F(real_A))
        BA = G_A(F(real_B))
        BB = G_B(F(real_B))

        AB=AB.data
        BA=BA.data


        if opt.dataPath == './datasets/cityscapes/val':
            vutils.save_image(AB,
                        opt.outf + 'cityscapes/leftImg8bit/%s/%s.png' % (find(listA[i]),replace('gtFine_color.png','leftImg8bit',listA[i])),
                        normalize=True)
            vutils.save_image(BA,
                        opt.outf + 'cityscapes/gtFine/%s/%s.png' % (find(listB[i]),replace('leftImg8bit.png','gtFine_color',listB[i])),
                        normalize=True)
        else:
            vutils.save_image(AB,
                        opt.outf + 'AB/%s.png' % (listA[i]),
                        normalize=True)
            vutils.save_image(BA,
                        opt.outf + 'BA/%s.png' % (listB[i]),
                        normalize=True)

test()

