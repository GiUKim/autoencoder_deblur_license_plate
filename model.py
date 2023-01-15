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

def ConvMixer(dim, depth, kernel_size=3, patch_size=5, n_classes=1000):
    return nn.Sequential(
        nn.Conv2d(init_ker, dim, kernel_size=patch_size, stride=patch_size),
        nn.GELU(),
        nn.BatchNorm2d(dim),
        *[nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size, groups=dim, padding=1),
                    nn.GELU(),
                    nn.BatchNorm2d(dim)
                )),
                nn.Conv2d(dim, dim, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(dim)
        ) for i in range(depth)],
    )

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

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
            self.pool = nn.AvgPool2d((2, 2))

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

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        if config.isColor:
            self.cbr1 = ConvBlock(pan_in=3, pan_out=16)
        else:
            self.cbr1 = ConvBlock(pan_in=1, pan_out=16)
        self.att1 = SE_Block(16, 4)
        self.cbr2 = ConvBlock(pan_in=16, pan_out=32)
        self.att2 = SE_Block(32, 4)
        self.cbr3 = ConvBlock(pan_in=32, pan_out=32, is_pool=True)
        self.att0 = SE_Block(32, 4)
        self.cbr4 = ConvBlock(pan_in=32, pan_out=32)
        self.att4 = SE_Block(32, 4)
        self.cbr5 = ConvBlock(pan_in=32, pan_out=64, is_pool=True)
        self.att00 = SE_Block(64, 4)
        self.cbr6 = ConvBlock(pan_in=64, pan_out=128)
        self.att6 = SE_Block(128, 4)
        self.cbr7 = ConvBlock(pan_in=128, pan_out=128, is_pool=True)
        self.cbr8 = ConvBlock(pan_in=128, pan_out=128)
        #self.att8 = SE_Block(128, 4)
        self.cbr9 = ConvBlock(pan_in=128, pan_out=128)
        self.conv10 = nn.Conv2d(128, config.num_classes, (1, 1))
        self.out_activation = nn.Sigmoid()
        self.output = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.cbr1(x)
        x = self.att1(x)
        x = self.cbr2(x)
        x = self.att2(x)
        x = self.cbr3(x)
        x = self.att0(x)
        x = self.cbr4(x)
        x = self.att4(x)
        x = self.cbr5(x)
        x = self.att00(x)
        x = self.cbr6(x)
        x = self.att6(x)
        x = self.cbr7(x)
        x = self.cbr8(x)
        x = self.cbr9(x)
        x = self.conv10(x)
        x = self.out_activation(x)
        out = self.output(x).squeeze()
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


class Per_patch_Fully_connected(nn.Module) :
    def __init__(self, input_size, patch_size, C) :
        super(Per_patch_Fully_connected, self).__init__()

        self.S = int((input_size[-2] * input_size[-1]) / (patch_size ** 2))
        self.x_dim_1_val = input_size[-3] * patch_size * patch_size
        self.projection_layer = nn.Linear(input_size[-3] * patch_size * patch_size,  C)

    def forward(self, x) :
        x = torch.reshape(x, (-1, self.S, self.x_dim_1_val))
        return self.projection_layer(x)

class token_mixing_MLP(nn.Module):
    def __init__(self, input_size):
        super(token_mixing_MLP, self).__init__()

        self.Layer_Norm = nn.LayerNorm(input_size[-2])  # C개의 값(columns)에 대해 각각 normalize 수행하므로 normalize되는 벡터의 크기는 S다.
        self.MLP = nn.Sequential(
            nn.Linear(input_size[-2], input_size[-2]),
            nn.GELU(),
            nn.Linear(input_size[-2], input_size[-2])
        )

    def forward(self, x):
        # layer_norm + transpose

        # [S x C]에서 column들을 가지고 연산하니까 Pytorch의 Layer norm을 적용하려면 transpose 하고 적용해야함.
        output = self.Layer_Norm(x.transpose(2, 1))  # transpose 후 Layer norm -> [C x S] 크기의 벡터가 나옴
        output = self.MLP(output)

        # [Batch x S x C] 형태로 transpose + skip connection
        output = output.transpose(2, 1)

        return output + x

class channel_mixing_MLP(nn.Module):
    def __init__(self, input_size):  #
        super(channel_mixing_MLP, self).__init__()

        self.Layer_Norm = nn.LayerNorm(input_size[-1])  # S개의 벡터를 가지고 각각 normalize하니까 normalize되는 벡터의 크기는 C다

        self.MLP = nn.Sequential(
            nn.Linear(input_size[-1], input_size[-1]),
            nn.GELU(),
            nn.Linear(input_size[-1], input_size[-1])
        )

    def forward(self, x):
        output = self.Layer_Norm(x)
        output = self.MLP(output)

        return output + x

class Mixer_Layer(nn.Module) :
    def __init__(self, input_size) : #
        super(Mixer_Layer, self).__init__()

        self.mixer_layer = nn.Sequential(
            token_mixing_MLP(input_size),
            channel_mixing_MLP(input_size)
        )
    def forward(self, x) :
        return self.mixer_layer(x)

class MLP_Mixer(nn.Module):
    def __init__(self, input_size, patch_size, C, N, classes_num):
        super(MLP_Mixer, self).__init__()

        S = int((input_size[-2] * input_size[-1]) / (patch_size ** 2))  # embedding으로 얻은 token의 개수

        self.mlp_mixer = nn.Sequential(
            Per_patch_Fully_connected(input_size, patch_size, C)
        )
        for i in range(N):  # Mixer Layer를 N번 쌓아준다
            self.mlp_mixer.add_module("Mixer_Layer_" + str(i), Mixer_Layer((S, C)))

        # Glboal Average Pooling
        # Appendix E에 pseudo code가 있길래 그거 보고 제작
        # LayerNorm 하고 token별로 평균을 구한다
        self.global_average_Pooling_1 = nn.LayerNorm([S, C])

        self.head = nn.Sequential(
            nn.Linear(S, classes_num),
            nn.Sigmoid()
        )
    def forward(self, x):
        if len(x.size()) == 3:
            x = torch.unsqueeze(x, 0)  # 4차원으로 늘려줌.
        output = self.mlp_mixer(x)
        output = self.global_average_Pooling_1(output)
        output = torch.mean(output, 2)
        return self.head(output)


############
from torchvision.models import resnet18, resnext50_32x4d
# ----- resnet 18 use -> size(224, 224, isColor=True 자동 수정), grad-cam(cbr9->layer4 로 쟈동 수정됨) #
def call_resnet18():
    model = resnet18(pretrained=True)
    model.fc = nn.Sequential(
        nn.Linear(512, config.num_classes),
        nn.Sigmoid()
       )
    for param in model.fc.parameters():
        param.requires_grad = True
    for name, param in model.named_parameters():
        print(name, param.requires_grad)
    return model

###########
def call_resnext50_32x4d():
    model = resnext50_32x4d(pretrained=True)
    model.fc = nn.Sequential(
            nn.Linear(2048, config.num_classes),
            nn.Sigmoid()
        )
    for param in model.fc.parameters():
        param.requires_grad = True
    for name, paramj in model.named_parameters():
        print(name, param.requires_grad)
    return model
