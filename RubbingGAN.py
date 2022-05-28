from __future__ import print_function
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from misc import *
import RubbingNet as net
from tensorboardX import SummaryWriter
from torch.autograd import Variable
import torchvision.utils as vutils
import torch.optim as optim
import argparse
from distutils.file_util import write_file
import os
import sys
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cudnn.fastest = True


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=False,
                    default='rubbing',  help='')
parser.add_argument('--dataroot', required=False,
                    default='', help='path to train dataset')
parser.add_argument('--valDataroot', required=False,
                    default='', help='path to val dataset')
parser.add_argument('--mode', type=str, default='A2B',
                    help='B2A: facade, A2B: edges2shoes')
parser.add_argument('--batchSize', type=int,
                    default=1, help='input batch size')
parser.add_argument('--valBatchSize', type=int,
                    default=64, help='input batch size')
parser.add_argument('--originalSize', type=int,
                    default=256, help='the height / width of the original input image')
parser.add_argument('--imageSize', type=int,
                    default=256, help='the height / width of the cropped input image to network')
parser.add_argument('--inputChannelSize', type=int,
                    default=3, help='size of the input channels')
parser.add_argument('--outputChannelSize', type=int,
                    default=3, help='size of the output channels')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=50,
                    help='number of epochs to train for')
parser.add_argument('--lrD', type=float, default=0.0002,
                    help='learning rate, default=0.0002')
parser.add_argument('--lrG', type=float, default=0.0002,
                    help='learning rate, default=0.0002')
parser.add_argument('--annealStart', type=int, default=0,
                    help='annealing learning rate start to')
parser.add_argument('--annealEvery', type=int, default=400,
                    help='epoch to reaching at learning rate of 0')
parser.add_argument('--lambdaGAN', type=float, default=1, help='lambdaGAN')
parser.add_argument('--lambdaIMG', type=float, default=0.1, help='lambdaIMG')
parser.add_argument('--poolSize', type=int, default=50,
                    help='Buffer size for storing previously generated samples from G')
parser.add_argument('--lambda_k', type=float,
                    default=0.001, help='learning rate of k')
parser.add_argument('--gamma', type=float, default=0.7,
                    help='balance bewteen D and G')
parser.add_argument('--wd', type=float, default=0.0000,
                    help='weight decay in D')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
parser.add_argument('--netG', default='', help="")
parser.add_argument('--netD', default='',
                    help="path to netD (to continue training)")
parser.add_argument('--workers', type=int,
                    help='number of data loading workers', default=0)
parser.add_argument('--exp', default='',
                    help='folder to output images and model checkpoints')
parser.add_argument('--display', type=int, default=200,
                    help='interval for displaying train-logs')
parser.add_argument('--evalIter', type=int, default=500,
                    help='interval for evauating(generating) images from valDataroot')
parser.add_argument('--hidden_size', type=int, default=64,
                    help='bottleneck dimension of Discriminator')
parser.add_argument('--log', default='', help="path to the log")
opt = parser.parse_args()
print(opt)

create_exp_dir(opt.exp)

opt.manualSeed = 101
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
torch.cuda.manual_seed_all(opt.manualSeed)
print("Random Seed: ", opt.manualSeed)

writeFile = opt.log
print("Log File:", writeFile)

writer1 = SummaryWriter(writeFile)
# get dataloader
dataloader = getLoader(opt.dataset,
                       opt.dataroot,
                       opt.originalSize,
                       opt.imageSize,
                       opt.batchSize,
                       opt.workers,
                       mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
                       split='train',
                       shuffle=True,
                       seed=opt.manualSeed)
valDataloader = getLoader(opt.dataset,
                          opt.valDataroot,
                          opt.imageSize,  # opt.originalSize,
                          opt.imageSize,
                          opt.valBatchSize,
                          opt.workers,
                          mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
                          split='val',
                          shuffle=False,
                          seed=opt.manualSeed)

# get logger
trainLogger = open('%s/train.log' % opt.exp, 'w')

ngf = opt.ngf
ndf = opt.ndf
inputChannelSize = opt.inputChannelSize
outputChannelSize = opt.outputChannelSize

# get models
netG = net.G(inputChannelSize, outputChannelSize, ngf)
netG.apply(weights_init)

if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)
netD = net.D(inputChannelSize + outputChannelSize, ndf)
netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

netD1 = net.D1(inputChannelSize, ndf, opt.hidden_size)
netD1.apply(weights_init)
print(netD1)
netG.train()
netD.train()
netD1.train()
criterionBCE = nn.BCELoss()
criterionCAE = nn.L1Loss()

target = torch.FloatTensor(
    opt.batchSize, outputChannelSize, opt.imageSize, opt.imageSize)
input = torch.FloatTensor(
    opt.batchSize, inputChannelSize, opt.imageSize, opt.imageSize)
val_target = torch.FloatTensor(
    opt.valBatchSize, outputChannelSize, opt.imageSize, opt.imageSize)
val_input = torch.FloatTensor(
    opt.valBatchSize, inputChannelSize, opt.imageSize, opt.imageSize)
label_d = torch.FloatTensor(opt.batchSize)
# NOTE: size of 2D output maps in the discriminator
sizePatchGAN = 30
real_label = 1
fake_label = 0

# image pool storing previously generated samples from G
imagePool = ImagePool(opt.poolSize)

# NOTE weight for L_cGAN and L_L1 (i.e. Eq.(4) in the paper)
lambdaGAN = opt.lambdaGAN
lambdaIMG = opt.lambdaIMG

netD.cuda()
netG.cuda()
netD1.cuda()
criterionBCE.cuda()
criterionCAE.cuda()
target, input, label_d = target.cuda(), input.cuda(), label_d.cuda()
val_target, val_input = val_target.cuda(), val_input.cuda()

target = Variable(target)
input = Variable(input)
label_d = Variable(label_d)

# get randomly sampled validation images and save it
val_iter = iter(valDataloader)
data_val = val_iter.next()
if opt.mode == 'B2A':
    val_target_cpu, val_input_cpu = data_val
elif opt.mode == 'A2B':
    val_input_cpu, val_target_cpu = data_val
val_target_cpu, val_input_cpu = val_target_cpu.cuda(), val_input_cpu.cuda()
val_target.resize_as_(val_target_cpu).copy_(val_target_cpu)
val_input.resize_as_(val_input_cpu).copy_(val_input_cpu)
vutils.save_image(val_target, '%s/real_target.png' % opt.exp, normalize=True)
vutils.save_image(val_input, '%s/real_input.png' % opt.exp, normalize=True)

# get optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lrD,
                        betas=(opt.beta1, 0.999), weight_decay=opt.wd)
optimizerG = optim.Adam(netG.parameters(), lr=opt.lrG,
                        betas=(opt.beta1, 0.999), weight_decay=0.0)
optimizerD1 = optim.Adam(netD1.parameters(), lr=opt.lrD, betas=(
    opt.beta1, 0.999), weight_decay=opt.wd)
# NOTE training loop
ganIterations = 0
k = 0  # control how much emphasis is put on L(G(z_D)) during gradient descent.
M_global = AverageMeter()
for epoch in range(opt.niter):

    if epoch > opt.annealStart:
        adjust_learning_rate(optimizerD, opt.lrD, epoch, None, opt.annealEvery)
        adjust_learning_rate(optimizerG, opt.lrG, epoch, None, opt.annealEvery)
        adjust_learning_rate(optimizerD1, opt.lrD, epoch,
                             None, opt.annealEvery)
    for i, data in enumerate(dataloader, 0):
        if opt.mode == 'B2A':
            target_cpu, input_cpu = data
        elif opt.mode == 'A2B':
            input_cpu, target_cpu = data

        batch_size = target_cpu.size(0)
        target_cpu, input_cpu = target_cpu.cuda(), input_cpu.cuda()
        # NOTE paired samples
        target.resize_as_(target_cpu).copy_(target_cpu)
        input.resize_as_(input_cpu).copy_(input_cpu)

        # max_D first
        for p in netD.parameters():
            p.requires_grad = True
        netD.zero_grad()

        # NOTE: compute L_cGAN in eq.(2)
        label_d.resize_((batch_size, 1, sizePatchGAN,
                        sizePatchGAN)).fill_(real_label)
        output = netD(torch.cat([target, input], 1))  # conditional
        errD_real = criterionBCE(output, label_d)
        errD_real.backward()
        D_x = output.data.mean()
        x_hat = netG(input)
        fake = x_hat.detach()
        fake = Variable(imagePool.query(fake.data))
        label_d.data.fill_(fake_label)
        output = netD(torch.cat([fake, input], 1))  # conditional
        errD_fake = criterionBCE(output, label_d)
        errD_fake.backward()

        errD = errD_real + errD_fake
        optimizerD.step()  # update Discriminator parameters

        # prevent computing gradients of weights in Discriminator
        for p in netD.parameters():
            p.requires_grad = False
        netG.zero_grad()  # start to update G

        # compute L_L1 (eq.(4) in the paper
        L_img_ = criterionCAE(x_hat, target)
        L_img = lambdaIMG * L_img_
        if lambdaIMG != 0:
            # in case of current version of pytorch
            L_img.backward(retain_graph=True)

        # compute L_cGAN (eq.(2) in the paper
        label_d.data.fill_(real_label)
        output = netD(torch.cat([x_hat, input], 1))
        errG_ = criterionBCE(output, label_d)
        errG = lambdaGAN * errG_
        if lambdaGAN != 0:
            errG.backward()   # update Generator parameters
        optimizerG.step()

        ####
        # max_D1 first
        for p in netD1.parameters():
            p.requires_grad = True
        netD1.zero_grad()
        # NOTE: compute L_D
        recon_real1 = netD1(target)
        x_hat1 = netG(input)
        fake1 = x_hat1.detach()
        # sample from image buffer
        fake1 = Variable(imagePool.query(fake1.data))
        recon_fake1 = netD1(fake1)
        # compute L(x,D) = |x-D(x)|
        errD_real1 = torch.mean(torch.abs(recon_real1 - target))
        # compute L(G(z_D),D1) = |G(z)-D(G(z))|
        errD_fake1 = torch.mean(torch.abs(recon_fake1 - fake1))
        # compute L_D1 = L(x，D1)- L(G(z_D),D1)
        errD1 = errD_real1 - k * errD_fake1
        errD1.backward()
        optimizerD1.step()

        # prevent computing gradients of weights in Discriminator
        for p in netD1.parameters():
            p.requires_grad = False
        netG.zero_grad()  # start to update G

        recon_fake1 = netD1(x_hat1)  # reuse previously computed x_hat
        # compute L_G = |G(z) - D(G(z))|
        errG_1 = torch.mean(torch.abs(recon_fake1 - x_hat1))
        errG1 = lambdaGAN * errG_1
        if lambdaGAN != 0:
            errG1.backward()
        # update praams
        optimizerG.step()

        # NOTE compute k_t and M_global
        balance = (opt.gamma * errD_real1 - errD_fake1).item()
        k = min(max(k + opt.lambda_k * balance, 0), 1)
        measure = errD_real1.item() + np.abs(balance)
        M_global.update(measure, target.size(0))
        ganIterations += 1

        if ganIterations % opt.display == 0:
            print('[%d/%d][%d/%d] L_D: %f L_img: %f L_G: %f D(x): %f L_D1: %f L_G1: %f'
                  % (epoch, opt.niter, i, len(dataloader),
                     errD.item(), L_img.item(), errG.item(), D_x, errD1.item(), errG1.item()))
            sys.stdout.flush()
            writer1.add_scalar('Loss_D1', errD.item(), global_step=epoch)
            writer1.add_scalar('L_img', L_img.item(), global_step=epoch)
            writer1.add_scalar('Loss_G', errG.item(), global_step=epoch)
            writer1.add_scalar('Loss_D2', errD1.item(), global_step=epoch)
            writer1.add_scalar('D(x)', D_x, global_step=epoch)
            writer1.add_scalar('L_D1', errD1.item(), global_step=epoch)
            writer1.add_scalar('balance', balance, global_step=epoch)
            writer1.add_scalar('k', k, global_step=epoch)
            writer1.add_scalar('measure', measure, global_step=epoch)
            writer1.add_scalar('M_global', M_global.avg, global_step=epoch)
            writer1.add_scalar('L_G1', errG1.item(), global_step=epoch)
            trainLogger.write('%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n' %
                              (epoch, errD.item(), errG.item(), L_img.item(), D_x, errD1.item(), k, M_global.avg, balance, errG1.item()))
            trainLogger.flush()

        if ganIterations % opt.evalIter == 0:
            val_batch_output = torch.FloatTensor(val_input.size()).fill_(0)
            for idx in range(val_input.size(0)):
                single_img = val_input[idx, :, :, :].unsqueeze(0)
                with torch.no_grad():
                    val_inputv = Variable(single_img)
                x_hat_val = netG(val_inputv)
                val_batch_output[idx, :, :, :].copy_(x_hat_val.data.squeeze(0))
            vutils.save_image(val_batch_output, '%s/generated_epoch_%08d_iter%08d.png' %
                              (opt.exp, epoch, ganIterations), normalize=True)

    # do checkpointing
            torch.save(netG.state_dict(), '%s/netG_epoch_%08d_iter%08d.pth' %
                       (opt.exp, epoch, ganIterations))

trainLogger.close()
