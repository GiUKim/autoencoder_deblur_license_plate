import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from glob import glob
from PIL import Image
from config import Config
from dataset import *

if config.isColor:
    init_ker = 3
else:
    init_ker = 1

class SE_Block(nn.Module):
    def __init__(self, c, r=4):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)

class ConvBlock(nn.Module):
    def __init__(self, pan_in, pan_out, kernel_size=(3, 3), padding=1, is_pool=False):
        super().__init__()
        self.is_pool = is_pool
        self.conv_layer = nn.Conv2d(pan_in, pan_out, kernel_size, padding=padding, bias=False)
        self.batchnorm = nn.BatchNorm2d(pan_out)
        self.relu = nn.ReLU()
        if is_pool:
            #self.pool = nn.AvgPool2d((2, 2))
            self.pool = nn.MaxPool2d((2, 2))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_uniform_(m.weight.data)
    def forward(self, x):
        x = self.conv_layer(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        if self.is_pool:
            x = self.pool(x)
        return x

class Net2(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        if config.isColor:
            self.cbr1 = ConvBlock(pan_in=3, pan_out=32, kernel_size=(7, 7), padding=3, is_pool=True)
        else:
            self.cbr1 = ConvBlock(pan_in=1, pan_out=32, kernel_size=(7, 7), padding=3, is_pool=True) # 128x128x32
        self.cbr2 = ConvBlock(pan_in=32, pan_out=64, is_pool=True) # 64x64x64
        self.att2 = SE_Block(64, 4)
        self.cbr3 = ConvBlock(pan_in=64, pan_out=128, is_pool=True) # 32x32x128
        self.att3 = SE_Block(128, 4)
        self.cbr4 = ConvBlock(pan_in=128, pan_out=256, is_pool=True) # 32x32x256
        self.conv_1_1 = nn.Conv2d(256, 1, (1, 1)) # 32x32x1
        self.enc_activation = nn.Sigmoid()
        self.enc_output = nn.Flatten() # (16x16) flatten, [width // 2^(# of pool), height // 2^(# of pool)]

        self.dec1 = nn.Linear(config.width // 2 ** (4) * config.height // 2 ** (4), 32)
        self.dec1_act = nn.ReLU()
        self.dec2 = nn.Linear(32, config.width * config.height * 3)
        self.dec2_out_act = nn.Sigmoid()

    def forward(self, x):
        x = self.cbr1(x)
        x = self.cbr2(x)
        x = self.att2(x)
        x = self.cbr3(x)
        x = self.att3(x)
        x = self.cbr4(x)
        x = self.conv_1_1(x)
        x = self.enc_activation(x)
        x = self.enc_output(x)

        x = self.dec1(x)
        x = self.dec1_act(x)
        x = self.dec2(x)
        out = self.dec2_out_act(x).reshape(-1, 3, config.height, config.width)
        return out

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.enc1 = nn.Linear(config.width * config.height * 3, 256)
        self.dec1 = nn.Linear(256, config.width * config.height * 3)
    def forward(self, x):
        x = self.enc1(x)
        out = self.dec1(x).reshape(-1, 3, config.height, config.width)
        return out

class Net_deeper(nn.Module):
    def __init__(self):
        super(Net_deeper, self).__init__()
        if config.isColor:
            self.cbr1 = ConvBlock(pan_in=3, pan_out=16)
        else:
            self.cbr1 = ConvBlock(pan_in=1, pan_out=16)
        self.att1 = SE_Block(16, 4)
        self.cbr2 = ConvBlock(pan_in=16, pan_out=32)
        self.cbr2_1 = ConvBlock(pan_in=32, pan_out=32)
        self.att2 = SE_Block(32, 4)
        self.cbr3 = ConvBlock(pan_in=32, pan_out=32, is_pool=True)
        self.att0 = SE_Block(32, 4)
        self.cbr4 = ConvBlock(pan_in=32, pan_out=32)
        self.cbr4_1 = ConvBlock(pan_in=32, pan_out=32)
        self.att4 = SE_Block(32, 4)
        self.cbr5 = ConvBlock(pan_in=32, pan_out=64, is_pool=True)
        self.att00 = SE_Block(64, 4)
        self.cbr6 = ConvBlock(pan_in=64, pan_out=128)
        self.cbr6_1 = ConvBlock(pan_in=128, pan_out=256)
        self.att6 = SE_Block(256, 4)
        self.cbr7 = ConvBlock(pan_in=256, pan_out=512, is_pool=True)
        self.cbr8 = ConvBlock(pan_in=512, pan_out=512)
        #self.att8 = SE_Block(128, 4)
        self.cbr9 = ConvBlock(pan_in=512, pan_out=512)
        self.conv10 = nn.Conv2d(512, config.num_classes, (1, 1))
        self.out_activation = nn.Sigmoid()
        self.output = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.cbr1(x)
        x = self.att1(x)
        x = self.cbr2(x)
        x = self.cbr2_1(x)
        x = self.att2(x)
        x = self.cbr3(x)
        x = self.att0(x)
        x = self.cbr4(x)
        x = self.cbr4_1(x)
        x = self.att4(x)
        x = self.cbr5(x)
        x = self.att00(x)
        x = self.cbr6(x)
        x = self.cbr6_1(x)
        x = self.att6(x)
        x = self.cbr7(x)
        x = self.cbr8(x)
        x = self.cbr9(x)
        x = self.conv10(x)
        x = self.out_activation(x)
        out = self.output(x).squeeze()
        return out

