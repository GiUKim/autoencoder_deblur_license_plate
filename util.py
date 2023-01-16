from config import Config
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
import random
import torch
from glob import glob
import math
from PIL import Image
import cv2
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np 
config = Config()

def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
def get_gaussian_filter() :
    kernel = np.array(
                [[0, 0, 0, 0, 0],
                [0, 1.0/25.0, 2.0/25.0, 1.0/25.0, 0],
                [0, 2.0/25.0, 4.0/25.0, 2.0/25.0, 0],
                [0, 1.0/25.0, 2.0/25.0, 1.0/25.0, 0],
                [0, 0, 0, 0, 0]])
    return kernel

def get_laplacian_black_filter() :
    kernel = np.array(
                [[0, 0, 1.0 / 25.0, 0, 0],
                [0, 1.0 / 25.0, 26.0 / 25.0, 1.0 / 25.0, 0],
                [1.0 / 25.0, 26.0 / 25.0, -112.0 / 25.0, 26.0 / 25.0, 1.0 / 25.0],
                [0, 1.0 / 25.0, 26.0 / 25.0, 1.0 / 25.0, 0],
                [0, 0, 1.0 / 25.0, 0, 0]])
    return kernel

def transform_img_to_resize(img, img_width, img_height):
    height, width = img.shape[:2]
    if width > img_width or height > img_height:
        img = cv2.resize(img, (img_width, img_height), cv2.INTER_LINEAR)
    elif width < img_width or height < img_height:
        img = cv2.resize(img, (img_width, img_height), cv2.INTER_LINEAR)
    return img

def change_feature_image(img) :
    img = img * 255
    img = img.astype(np.uint8)
    if config.isColor:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img_denose = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    gray = cv2.cvtColor(img_denose, cv2.COLOR_RGB2GRAY)

    height, width = img.shape[:2]
    if config.width != width or config.height != height:
        gray = transform_img_to_resize(gray, config.width, config.height)
    kernel = get_gaussian_filter()
    gray = cv2.filter2D(gray, -1, kernel)
    kernel = get_laplacian_black_filter()
    binary1 = cv2.filter2D(gray, -1, kernel)
    t, t_otsu = cv2.threshold(binary1, -1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    contours, hierachy = cv2.findContours(t_otsu, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)
    img_result = gray.copy().astype('uint8')
    cv2.drawContours(img_result, contours, -1, (0, 0, 0), 1)
    return img_result , t_otsu

def apply_custom_aug(image, **kwargs):
    if config.isColor:
        h, w, _ = image.shape
    else:
        h, w = image.shape
    ret, _ = change_feature_image(image)
    if config.isColor:
        ret = cv2.cvtColor(ret, cv2.COLOR_GRAY2RGB)
    else:
        pass
    ret = cv2.resize(ret, (config.width, config.height))
    ret = ret / 255.
    ret = ret.astype(np.float32)
    return ret

def preprocess_image(img):
    if config.isColor:
        preprocessed_img = img.copy()[:, :, ::-1]
        preprocessed_img = np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    else:
        preprocessed_img = np.expand_dims(img, -1)
        preprocessed_img = np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = preprocessed_img.requires_grad_(True)
    return input

def threshold(x):
    mean_ = x.mean()
    std_ = x.std()
    thres = mean_ + std_
    x = (x > thres)
    return x

def normalize(Ac):
    Ac_shape = Ac.shape
    AA = Ac.view(Ac.size(0), -1)
    AA -= AA.min(1, keepdim=True)[0]
    AA /= AA.max(1, keepdim=True)[0]
    scaled_ac = AA.view(Ac_shape)
    return scaled_ac

def tensor2image(x, i=0):
    x = normalize(x)
    x = x[i].detach().cpu().numpy()
    x = np.transpose(x, (1, 2, 0))
    return x

def visualize_img(input_img, output_img):
    # input_img = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)
    # output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
    fig = plt.figure(figsize=(4, 4))
    plt.subplots_adjust(bottom=0.01)
    plt.axis('off')
    plt.subplot(2, 1, 1)
    plt.imshow(input_img)
    plt.subplot(2, 1, 2)
    plt.imshow(output_img)
    fig.canvas.draw()
    plt.close()
    display_img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    display_img = display_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    #display_img = cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR)
    cv2.imshow('Validation', display_img)
    cv2.waitKey(1)


class LRScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warm_up=0.1, decay=0.1, total_epoch=100, last_epoch=-1):
        def lr_lambda(step):
            step += 1
            print('\n', 'LR:', optimizer.param_groups[0]['lr'], '\n')
            warm_up_epoch = total_epoch * warm_up
            if step < total_epoch * warm_up:
                return ((math.cos(((step * math.pi) / warm_up_epoch) + math.pi) + 1.0) * 0.5) 
            elif total_epoch * 0.8 < step <= total_epoch * 0.9:
                return decay
            elif step > total_epoch * 0.9:
                return decay ** 2
            else:
                return 1.0
        super(LRScheduler, self).__init__(optimizer, lr_lambda, last_epoch=last_epoch)

import torch.nn as nn
class ALE_loss(nn.Module):
    def __init__(self, gamma=0.0):
        super(ALE_loss, self).__init__()
        self.gamma = gamma
    def forward(self, output, target):
        absval = torch.abs(output - target)
        return -(torch.log(1. - absval)) * (absval ** self.gamma)

