from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from itertools import chain

from utils.dataset import DATASET
from utils.ImagePool import ImagePool
from model.Discriminator import Discriminator
from model.Generator import Generator
from model.Encoder import Encoder

parser = argparse.ArgumentParser(description='train BrGAN model')
parser.add_argument('--batchSize', type=int, default=4, help='with batchSize=1 equivalent to instance normalization.')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=100000, help='number of iterations to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay in network D, default=1e-4')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--outf', default='checkpoints/cityscapes/', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--dataPath', default='./datasets/cityscapes/train/', help='path to training images')
parser.add_argument('--loadSize', type=int, default=143, help='scale image to this size')
parser.add_argument('--fineSize', type=int, default=128, help='random crop image to this size')
parser.add_argument('--flip', type=int, default=1, help='1 for flipping image randomly, 0 for not')
parser.add_argument('--input_nc', type=int, default=3, help='channel number of input image')
parser.add_argument('--output_nc', type=int, default=3, help='channel number of output image')
parser.add_argument('--G_A', default='', help='path to pre-trained G_A')
parser.add_argument('--G_B', default='', help='path to pre-trained G_B')
parser.add_argument('--save_step', type=int, default=10000, help='save interval')
parser.add_argument('--log_step', type=int, default=100, help='log interval')
parser.add_argument('--loss_type', default='mse', help='GAN loss type, bce|mse default is negative likelihood loss')
parser.add_argument('--poolSize', type=int, default=100, help='size of buffer in lsGAN, poolSize=0 indicates not using history')
parser.add_argument('--lambda_DAA', type=float, default=1.0, help='weight of adversarial loss DAA')
parser.add_argument('--lambda_DAB', type=float, default=1.0, help='weight of adversarial loss DAB')
parser.add_argument('--lambda_DBA', type=float, default=1.0, help='weight of adversarial loss DBA')
parser.add_argument('--lambda_DBB', type=float, default=1.0, help='weight of adversarial loss DBB')
parser.add_argument('--lambda_DABA', type=float, default=1.0, help='weight of adversarial loss DABA')
parser.add_argument('--lambda_DBAB', type=float, default=1.0, help='weight of adversarial loss DBAB')
parser.add_argument('--lambda_AA', type=float, default=8.0, help='weight of reconstruction loss AA')
parser.add_argument('--lambda_BB', type=float, default=8.0, help='weight of reconstruction loss BB')
parser.add_argument('--lambda_ABVV1', type=float, default=10.0, help='weight of encoding loss ABVV1')
parser.add_argument('--lambda_ABVV2', type=float, default=10.0, help='weight of encoding loss ABVV2')
opt = parser.parse_args()
print(opt)

subdict = "../best/"

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
                                       shuffle=True,
                                       num_workers=2)
loaderA = iter(loader_A)
loader_B = torch.utils.data.DataLoader(dataset=datasetB,
                                       batch_size=opt.batchSize,
                                       shuffle=True,
                                       num_workers=2)
loaderB = iter(loader_B)
ABPool = ImagePool(opt.poolSize)
BAPool = ImagePool(opt.poolSize)
###########   MODEL   ###########
# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

ndf = opt.ndf
ngf = opt.ngf
nc = 3

D_A = Discriminator(opt.input_nc,ndf)
D_B = Discriminator(opt.output_nc,ndf)
G_A = Generator(opt.input_nc, opt.output_nc, opt.ngf)
G_B = Generator(opt.output_nc, opt.input_nc, opt.ngf)
F = Encoder(opt.input_nc, opt.output_nc, opt.ngf)

if(opt.G_A != ''):
    print('Warning! Loading pre-trained weights.')
    G_A.load_state_dict(torch.load(opt.G_A))
    G_B.load_state_dict(torch.load(opt.G_B))
    F.load_state_dict(torch.load(opt.F))
else:
    G_A.apply(weights_init)
    G_B.apply(weights_init)
    F.apply(weights_init)

if(opt.cuda):
    D_A.cuda()
    D_B.cuda()
    G_A.cuda()
    G_B.cuda()
    F.cuda()


D_A.apply(weights_init)
D_B.apply(weights_init)

###########   LOSS & OPTIMIZER   ##########
criterionMSE = nn.L1Loss()
if(opt.loss_type == 'bce'):
    criterion = nn.BCELoss()
else:
    criterion = nn.MSELoss()
# chain is used to update two generators simultaneously
optimizerD_A = torch.optim.Adam(D_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
optimizerD_B = torch.optim.Adam(D_B.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
optimizerG_A = torch.optim.Adam(G_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
optimizerG_B = torch.optim.Adam(G_B.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
optimizerF = torch.optim.Adam(F.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)

###########   GLOBAL VARIABLES   ###########
input_nc = opt.input_nc
output_nc = opt.output_nc
fineSize = opt.fineSize

real_A = torch.FloatTensor(opt.batchSize, input_nc, fineSize, fineSize)
real_B = torch.FloatTensor(opt.batchSize, input_nc, fineSize, fineSize)
AA = torch.FloatTensor(opt.batchSize, input_nc, fineSize, fineSize)
AB = torch.FloatTensor(opt.batchSize, output_nc, fineSize, fineSize)
BA = torch.FloatTensor(opt.batchSize, input_nc, fineSize, fineSize)
BB = torch.FloatTensor(opt.batchSize, output_nc, fineSize, fineSize)
ABA = torch.FloatTensor(opt.batchSize, output_nc, fineSize, fineSize)
BAB = torch.FloatTensor(opt.batchSize, output_nc, fineSize, fineSize)
label = torch.FloatTensor(opt.batchSize)
print(fineSize)
ABV1 = torch.FloatTensor(opt.batchSize, ngf*4, fineSize//4, fineSize//4)
ABVV11 = torch.FloatTensor(opt.batchSize, ngf*4, fineSize//4, fineSize//4)
ABVV12 = torch.FloatTensor(opt.batchSize, ngf*4, fineSize//4, fineSize//4)
ABV2 = torch.FloatTensor(opt.batchSize, ngf*4, fineSize//4, fineSize//4)
ABVV21 = torch.FloatTensor(opt.batchSize, ngf*4, fineSize//4, fineSize//4)
ABVV22 = torch.FloatTensor(opt.batchSize, ngf*4, fineSize//4, fineSize//4)

AV_tmp = torch.FloatTensor(opt.batchSize, ngf*4, fineSize//4, fineSize//4)
BV_tmp = torch.FloatTensor(opt.batchSize, ngf*4, fineSize//4, fineSize//4)


real_A = Variable(real_A)
real_B = Variable(real_B)
label = Variable(label)
AA = Variable(AA)
AB = Variable(AB)
BA = Variable(BA)
BB = Variable(BB)
ABA = Variable(ABA)
BAB = Variable(BAB)
ABV1 = Variable(ABV1)
ABVV11 = Variable(ABVV11)
ABVV12 = Variable(ABVV12)
ABV2 = Variable(ABV2)
ABVV21 = Variable(ABVV21)
ABVV22 = Variable(ABVV22)

AV_tmp = Variable(AV_tmp)
BV_tmp = Variable(BV_tmp)

if(opt.cuda):
    real_A = real_A.cuda()
    real_B = real_B.cuda()
    label = label.cuda()
    AA = AA.cuda()
    AB = AB.cuda()
    BA = BA.cuda()
    BB = BB.cuda()
    ABA = ABA.cuda()
    BAB = BAB.cuda()
    ABV1 = ABV1.cuda()
    ABVV11 = ABVV11.cuda()
    ABVV12 = ABVV12.cuda()
    ABV2 = ABV2.cuda()
    ABVV21 = ABVV21.cuda()
    ABVV22 = ABVV22.cuda()
    criterion.cuda()
    criterionMSE.cuda()

    AV_tmp = AV_tmp.cuda()
    BV_tmp = BV_tmp.cuda()

real_label = 1
fake_label = 0



###########   Training   ###########
D_A.train()
D_B.train()
G_A.train()
G_B.train()
F.train()


for iteration in range(1,opt.niter+1):

    if iteration % 10000 == 0:
        for param_group in optimizerD_A.param_groups:
            param_group['lr'] = param_group['lr'] * 0.9
        for param_group in optimizerD_B.param_groups:
            param_group['lr'] = param_group['lr'] * 0.9
        for param_group in optimizerG_A.param_groups:
            param_group['lr'] = param_group['lr'] * 0.9
        for param_group in optimizerG_B.param_groups:
            param_group['lr'] = param_group['lr'] * 0.9
        for param_group in optimizerF.param_groups:
            param_group['lr'] = param_group['lr'] * 0.9

    ###########   train discriminators  ###########
    for count in range (0,1):
        try:
            imgA = loaderA.next()
            imgB = loaderB.next()
        except StopIteration:
            loaderA, loaderB = iter(loader_A), iter(loader_B)
            imgA = loaderA.next()
            imgB = loaderB.next()

        real_A.data.resize_(imgA.size()).copy_(imgA)
        real_B.data.resize_(imgB.size()).copy_(imgB)


        D_A.zero_grad()
        D_B.zero_grad()

        # train with real
        outA = D_A(real_A)
        outB = D_B(real_B)
        label.data.resize_(outA.size())
        label.data.fill_(real_label)
        l_A = criterion(outA, label)
        l_B = criterion(outB, label)
        errD_real = l_A + l_B
        errD_real.backward()

        # train with fake
        label.data.fill_(fake_label)

        AB_tmp = G_B(F(real_A))
        AB.data.resize_(AB_tmp.data.size()).copy_(ABPool.Query(AB_tmp.cpu().data))
        BA_tmp = G_A(F(real_B))
        BA.data.resize_(BA_tmp.data.size()).copy_(BAPool.Query(BA_tmp.cpu().data))

        out_BA = D_A(BA.detach())
        out_AB = D_B(AB.detach())

        l_BA = criterion(out_BA,label)
        l_AB = criterion(out_AB,label)

        errD_fake = l_BA + l_AB
        errD_fake.backward()

        errD = (errD_real + errD_fake)*0.5
        optimizerD_A.step()
        optimizerD_B.step()

        errD_A = l_A + l_BA
        errD_B = l_B + l_AB



    ########### train single-encoder-dual-decoder (SEDD) ###########
    for count in range (0,3):
        try:
            imgA = loaderA.next()
            imgB = loaderB.next()
        except StopIteration:
            loaderA, loaderB = iter(loader_A), iter(loader_B)
            imgA = loaderA.next()
            imgB = loaderB.next()

        real_A.data.resize_(imgA.size()).copy_(imgA)
        real_B.data.resize_(imgB.size()).copy_(imgB)

        G_A.zero_grad()
        G_B.zero_grad()
        F.zero_grad()


        ABV1 = F(real_A)
        AA = G_A(ABV1)
        AB = G_B(ABV1)
        ABVV11 = F(AA)
        ABVV12 = F(AB)
        ABA = G_A(ABVV12)

        ABV2 = F(real_B)
        BA = G_A(ABV2)
        BB = G_B(ABV2)
        ABVV21 = F(BA)
        ABVV22 = F(BB)
        BAB = G_B(ABVV21)

        out_AA = D_A(AA)
        out_AB = D_B(AB)
        out_BA = D_A(BA)
        out_BB = D_B(BB)
        out_ABA = D_A(ABA)
        out_BAB = D_B(BAB)

        label.data.resize_(out_AA.size())
        label.data.fill_(real_label)

        # adversarial loss
        l_AA = criterion(out_AA,label) * opt.lambda_DAA
        l_AB = criterion(out_AB,label) * opt.lambda_DAB
        l_BA = criterion(out_BA,label) * opt.lambda_DBA
        l_BB = criterion(out_BB,label) * opt.lambda_DBB
        l_ABA = criterion(out_ABA,label) * opt.lambda_DABA
        l_BAB = criterion(out_BAB,label) * opt.lambda_DBAB

        # reconstruction loss
        l_rec_AA = criterionMSE(AA, real_A) * opt.lambda_AA
        l_rec_BB = criterionMSE(BB, real_B) * opt.lambda_BB

        # encoding loss
        tmp_ABV1 = ABV1.detach()
        tmp_ABV2 = ABV2.detach()
        l_rec_ABVV1 = criterionMSE(ABVV11, tmp_ABV1) * opt.lambda_ABVV1 + criterionMSE(ABVV22, tmp_ABV2) * opt.lambda_ABVV1 #ABVV1: feature map of AA and BB
        l_rec_ABVV2 = criterionMSE(ABVV21, tmp_ABV2) * opt.lambda_ABVV2 + criterionMSE(ABVV12, tmp_ABV1) * opt.lambda_ABVV2 #ABVV2: feature map of AB and BA

        # full objective
        errGAN = l_AA + l_AB + l_BA + l_BB + l_ABA + l_BAB
        errMSE =  l_rec_AA + l_rec_BB + l_rec_ABVV1 + l_rec_ABVV2
        errG = errGAN + errMSE

        # backward
        errG.backward()
        optimizerG_A.step()
        optimizerG_B.step()
        optimizerF.step()


    ###########   Logging   ############
    print('[%d/%d] Loss_D: %.4f Loss_G: %.4f Loss_MSE: %.4f'
        % (iteration, opt.niter,
            errD.data[0], errGAN.data[0], errMSE.data[0]))



    if iteration % opt.save_step == 0:
        torch.save(G_A.state_dict(), '{}/G_A_{}.pth'.format(opt.outf, iteration))
        torch.save(G_B.state_dict(), '{}/G_B_{}.pth'.format(opt.outf, iteration))
        torch.save(F.state_dict(), '{}/F_{}.pth'.format(opt.outf, iteration))