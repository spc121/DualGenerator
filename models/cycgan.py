import os
import glob
import random
import torch
import itertools
import datetime
import time
import sys
import argparse
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
from torchvision.utils import save_image, make_grid



#########################################################################
############################  models  ###################################

## 定义参数初始化函数
def weights_init_normal(m):
    classname = m.__class__.__name__  ## m作为一个形参，原则上可以传递很多的内容, 为了实现多实参传递，每一个moudle要给出自己的name. 所以这句话就是返回m的名字.
    if classname.find("Conv") != -1:  ## find():实现查找classname中是否含有Conv字符，没有返回-1；有返回0.
        torch.nn.init.normal_(m.weight.data, 0.0,
                              0.02)  ## m.weight.data表示需要初始化的权重。nn.init.normal_():表示随机初始化采用正态分布，均值为0，标准差为0.02.
        if hasattr(m, "bias") and m.bias is not None:  ## hasattr():用于判断m是否包含对应的属性bias, 以及bias属性是否不为空.
            torch.nn.init.constant_(m.bias.data, 0.0)  ## nn.init.constant_():表示将偏差定义为常量0.
    elif classname.find("BatchNorm2d") != -1:  ## find():实现查找classname中是否含有BatchNorm2d字符，没有返回-1；有返回0.
        torch.nn.init.normal_(m.weight.data, 1.0,
                              0.02)  ## m.weight.data表示需要初始化的权重. nn.init.normal_():表示随机初始化采用正态分布，均值为0，标准差为0.02.
        torch.nn.init.constant_(m.bias.data, 0.0)  ## nn.init.constant_():表示将偏差定义为常量0.


##############################
##  残差块儿ResidualBlock
##############################
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(  ## block = [pad + conv + norm + relu + pad + conv + norm]
            nn.ReflectionPad2d(1),  ## ReflectionPad2d():利用输入边界的反射来填充输入张量
            nn.Conv2d(in_features, in_features, 3),  ## 卷积
            nn.InstanceNorm2d(in_features),  ## InstanceNorm2d():在图像像素上对HW做归一化，用在风格化迁移
            nn.ReLU(inplace=True),  ## 非线性激活
            nn.ReflectionPad2d(1),  ## ReflectionPad2d():利用输入边界的反射来填充输入张量
            nn.Conv2d(in_features, in_features, 3),  ## 卷积
            nn.InstanceNorm2d(in_features),  ## InstanceNorm2d():在图像像素上对HW做归一化，用在风格化迁移
        )

    def forward(self, x):  ## 输入为 一张图像
        return x + self.block(x)  ## 输出为 图像加上网络的残差输出


##############################
##  生成器网络GeneratorResNet
##############################
class GeneratorResNet(nn.Module):
    def __init__(self, input_shape, num_residual_blocks):  ## (input_shape = (3, 256, 256), num_residual_blocks = 9)
        super(GeneratorResNet, self).__init__()

        channels = input_shape[0]  ## 输入通道数channels = 3

        ## 初始化网络结构
        out_features = 64  ## 输出特征数out_features = 64
        model = [  ## model = [Pad + Conv + Norm + ReLU]
            nn.ReflectionPad2d(channels),  ## ReflectionPad2d(3):利用输入边界的反射来填充输入张量
            nn.Conv2d(channels, out_features, 7),  ## Conv2d(3, 64, 7)
            nn.InstanceNorm2d(out_features),  ## InstanceNorm2d(64):在图像像素上对HW做归一化，用在风格化迁移
            nn.ReLU(inplace=True),  ## 非线性激活
        ]
        in_features = out_features  ## in_features = 64

        ## 下采样，循环2次
        for _ in range(2):
            out_features *= 2  ## out_features = 128 -> 256
            model += [  ## (Conv + Norm + ReLU) * 2
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features  ## in_features = 256

        # 残差块儿，循环9次
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(out_features)]  ## model += [pad + conv + norm + relu + pad + conv + norm]

        # 上采样两次
        for _ in range(2):
            out_features //= 2  ## out_features = 128 -> 64
            model += [  ## model += [Upsample + conv + norm + relu]
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features  ## out_features = 64

        ## 网络输出层                                                            ## model += [pad + conv + tanh]
        model += [nn.ReflectionPad2d(channels), nn.Conv2d(out_features, channels, 7),
                  nn.Tanh()]  ## 将(3)的数据每一个都映射到[-1, 1]之间

        self.model = nn.Sequential(*model)

    def forward(self, x):  ## 输入(1, 3, 256, 256)
        return self.model(x)  ## 输出(1, 3, 256, 256)


##############################
#        Discriminator
##############################
class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        channels, height, width = input_shape  ## input_shape:(3， 256， 256)

        # Calculate output shape of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)  ## output_shape = (1, 16, 16)

        def discriminator_block(in_filters, out_filters, normalize=True):  ## 鉴别器块儿
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]  ## layer += [conv + norm + relu]
            if normalize:  ## 每次卷积尺寸会缩小一半，共卷积了4次
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),  ## layer += [conv(3, 64) + relu]
            *discriminator_block(64, 128),  ## layer += [conv(64, 128) + norm + relu]
            *discriminator_block(128, 256),  ## layer += [conv(128, 256) + norm + relu]
            *discriminator_block(256, 512),  ## layer += [conv(256, 512) + norm + relu]
            nn.ZeroPad2d((1, 0, 1, 0)),  ## layer += [pad]
            nn.Conv2d(512, 1, 4, padding=1)  ## layer += [conv(512, 1)]
        )

    def forward(self, img):  ## 输入(1, 3, 256, 256)
        return self.model(img)  ## 输出(1, 1, 16, 16)


#########################################################################
############################  utils  ###################################

## 先前生成的样本的缓冲区
class ReplayBuffer:
    def __init__(self, max_size=50):
        assert max_size > 0, "Empty buffer or trying to create a black hole. Be careful."
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):  ## 放入一张图像，再从buffer里取一张出来
        to_return = []  ## 确保数据的随机性，判断真假图片的鉴别器识别率
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:  ## 最多放入50张，没满就一直添加
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:  ## 满了就1/2的概率从buffer里取，或者就用当前的输入图片
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))


# ## 设置学习率为初始学习率乘以给定lr_lambda函数的值
# class LambdaLR:
#     def __init__(self, n_epochs, offset, decay_start_epoch):  ## (n_epochs = 50, offset = epoch, decay_start_epoch = 30)
#         assert (
#                            n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"  ## 断言，要让n_epochs > decay_start_epoch 才可以
#         self.n_epochs = n_epochs
#         self.offset = offset
#         self.decay_start_epoch = decay_start_epoch
#
#     def step(self, epoch):  ## return    1-max(0, epoch - 30) / (50 - 30)
#         return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)


###########################################################################
############################  cycle_gan  ###################################

def train(fine, gt):
    # ----------
    #  Training
    # ----------
    fine = fine.permute(2, 1, 0).contiguous()
    gt = gt.permute(2, 1, 0).contiguous()
    fine_s = torch.cat((fine, fine), dim=2)
    gt_s = torch.cat((gt, gt), dim=2)
    fine_s = torch.cat((fine_s, fine_s), dim=2)
    gt_s = torch.cat((gt_s, gt_s), dim=2)
    fine_s = torch.cat((fine_s, fine_s), dim=2)
    gt_s = torch.cat((gt_s, gt_s), dim=2)
    fine_s = fine_s.view(1, 3, 128, 256)
    gt_s = gt_s.view(1, 3, 128, 256)
    fine_s = torch.cat((fine_s, fine_s), dim=2)
    gt_s = torch.cat((gt_s, gt_s), dim=2)

    ## input_shape:(3, 256, 256)

    input_shape = (3, 256, 256)

    ## 创建生成器，判别器对象
    G_AB = GeneratorResNet(input_shape, 9)
    G_BA = GeneratorResNet(input_shape, 9)
    D_A = Discriminator(input_shape)
    D_B = Discriminator(input_shape)


    ## 损失函数
    ## MES 二分类的交叉熵
    ## L1loss 相比于L2 Loss保边缘
    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()

    ## 如果有显卡，都在cuda模式中运行
    if torch.cuda.is_available():
        G_AB = G_AB.cuda()
        G_BA = G_BA.cuda()
        D_A = D_A.cuda()
        D_B = D_B.cuda()
        criterion_GAN.cuda()
        criterion_cycle.cuda()
        criterion_identity.cuda()

    # if opt.epoch != 0:
    #     # 载入训练到第n轮的预训练模型
    #     G_AB.load_state_dict(torch.load("save/%s/G_AB_%d.pth" % (opt.dataset_name, opt.epoch)))
    #     G_BA.load_state_dict(torch.load("save/%s/G_BA_%d.pth" % (opt.dataset_name, opt.epoch)))
    #     D_A.load_state_dict(torch.load("save/%s/D_A_%d.pth" % (opt.dataset_name, opt.epoch)))
    #     D_B.load_state_dict(torch.load("save/%s/D_B_%d.pth" % (opt.dataset_name, opt.epoch)))
    # else:
    #     # 初始化模型参数
    #     G_AB.apply(weights_init_normal)
    #     G_BA.apply(weights_init_normal)
    #     D_A.apply(weights_init_normal)
    #     D_B.apply(weights_init_normal)

    ## 定义优化函数,优化函数的学习率为0.0003
    optimizer_G = torch.optim.Adam(
        itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=0.0003, betas=(0.5, 0.999)
    )
    optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=0.0003, betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=0.0003, betas=(0.5, 0.999))


    ## 先前生成的样本的缓冲区
    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    ## 读取数据集中的真图片
    ## 将tensor变成Variable放入计算图中，tensor变成variable之后才能进行反向传播求梯度
    real_A = Variable(fine_s).cuda()  ## 真图像A
    real_B = Variable(gt_s).cuda()  ## 真图像B

    ## 全真，全假的标签
    valid = Variable(torch.ones((real_A.size(0), *D_A.output_shape)),
                     requires_grad=False).cuda()  ## 定义真实的图片label为1 ones((1, 1, 16, 16))
    fake = Variable(torch.zeros((real_A.size(0), *D_A.output_shape)),
                    requires_grad=False).cuda()  ## 定义假的图片的label为0 zeros((1, 1, 16, 16))

    ## -----------------
    ##  Train Generator
    ## 原理：目的是希望生成的假的图片被判别器判断为真的图片，
    ## 在此过程中，将判别器固定，将假的图片传入判别器的结果与真实的label对应，
    ## 反向传播更新的参数是生成网络里面的参数，
    ## 这样可以通过更新生成网络里面的参数，来训练网络，使得生成的图片让判别器以为是真的, 这样就达到了对抗的目的
    ## -----------------
    G_AB.train()
    G_BA.train()

    ## Identity loss                                              ## A风格的图像 放在 B -> A 生成器中，生成的图像也要是 A风格
    loss_id_A = criterion_identity(G_BA(real_A),
                                   real_A)  ## loss_id_A就是把图像A1放入 B2A 的生成器中，那当然生成图像A2的风格也得是A风格, 要让A1,A2的差距很小
    loss_id_B = criterion_identity(G_AB(real_B), real_B)

    loss_identity = (loss_id_A + loss_id_B) / 2  ## Identity loss

    ## GAN loss
    fake_B = G_AB(real_A)  ## 用真图像A生成的假图像B
    loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)  ## 用B鉴别器鉴别假图像B，训练生成器的目的就是要让鉴别器以为假的是真的，假的太接近真的让鉴别器分辨不出来
    fake_A = G_BA(real_B)  ## 用真图像B生成的假图像A
    loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)  ## 用A鉴别器鉴别假图像A，训练生成器的目的就是要让鉴别器以为假的是真的,假的太接近真的让鉴别器分辨不出来

    loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2  ## GAN loss

    # Cycle loss 循环一致性损失
    recov_A = G_BA(fake_B)  ## 之前中realA 通过 A -> B 生成的假图像B，再经过 B -> A ，使得fakeB 得到的循环图像recovA，
    loss_cycle_A = criterion_cycle(recov_A, real_A)  ## realA和recovA的差距应该很小，以保证A,B间不仅风格有所变化，而且图片对应的的细节也可以保留
    recov_B = G_AB(fake_A)
    loss_cycle_B = criterion_cycle(recov_B, real_B)

    loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

    # Total loss                                                  ## 就是上面所有的损失都加起来
    loss_G = loss_GAN + 10 * loss_cycle + 5 * loss_identity
    optimizer_G.zero_grad()  ## 在反向传播之前，先将梯度归0
    loss_G.backward()  ## 将误差反向传播
    optimizer_G.step()  ## 更新参数

    ## -----------------------
    ## Train Discriminator A
    ## 分为两部分：1、真的图像判别为真；2、假的图像判别为假
    ## -----------------------
    ## 真的图像判别为真
    loss_real = criterion_GAN(D_A(real_A), valid)
    ## 假的图像判别为假(从之前的buffer缓存中随机取一张)
    fake_A_ = fake_A_buffer.push_and_pop(fake_A)
    loss_fake = criterion_GAN(D_A(fake_A_.detach()), fake)
    # Total loss
    loss_D_A = (loss_real + loss_fake) / 2
    optimizer_D_A.zero_grad()  ## 在反向传播之前，先将梯度归0
    loss_D_A.backward()  ## 将误差反向传播
    optimizer_D_A.step()  ## 更新参数

    ## -----------------------
    ## Train Discriminator B
    ## 分为两部分：1、真的图像判别为真；2、假的图像判别为假
    ## -----------------------
    # 真的图像判别为真
    loss_real = criterion_GAN(D_B(real_B), valid)
    ## 假的图像判别为假(从之前的buffer缓存中随机取一张)
    fake_B_ = fake_B_buffer.push_and_pop(fake_B)
    loss_fake = criterion_GAN(D_B(fake_B_.detach()), fake)
    # Total loss
    loss_D_B = (loss_real + loss_fake) / 2
    optimizer_D_B.zero_grad()  ## 在反向传播之前，先将梯度归0
    loss_D_B.backward()  ## 将误差反向传播
    optimizer_D_B.step()  ## 更新参数

    loss_D = (loss_D_A + loss_D_B) / 2


    return loss_D, loss_G

