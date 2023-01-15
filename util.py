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

def print_validation_eval_log(correct_list):
    for _idx, cls_correct in enumerate(correct_list):
        if not config.has_unknown:
            if _idx == len(config.class_list):
                break
            print('Class {:>2}: {:<11} Accuracy: {:>8}/{:<8} ({:.2f}%)'.format(
                _idx, config.class_list[_idx], correct_list[_idx], config.test_class_each_num[_idx],
                100. * (correct_list[_idx] / config.test_class_each_num[_idx])
            )
            )
        else:
            if _idx == len(config.class_list) - 1:  # unknown acc
                print('Class {:>2}: {:<11} Accuracy: {:>8}/{:<8} ({:.2f}%)'.format(
                    'N', "unknown", correct_list[-1], config.test_class_each_num[config.unknown_idx],
                    100. * (correct_list[-1] / config.test_class_each_num[config.unknown_idx])
                )
                )
            elif _idx >= config.unknown_idx and config.unknown_idx != len(config.class_list) - 1:
                print('Class {:>2}: {:<11} Accuracy: {:>8}/{:<8} ({:.2f}%)'.format(
                    _idx, config.class_list[_idx + 1], correct_list[_idx], config.test_class_each_num[_idx + 1],
                    100. * (correct_list[_idx] / config.test_class_each_num[_idx + 1])
                )
                )
            else:
                print('Class {:>2}: {:<11} Accuracy: {:>8}/{:<8} ({:.2f}%)'.format(
                    _idx, config.class_list[_idx], cls_correct, config.test_class_each_num[_idx],
                    100. * (cls_correct / config.test_class_each_num[_idx])
                )
                )
    print('=' * 80)

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


def get_display_class_num():
    if len(config.class_list_without_unknown) < 5:
        display_class_num = len(config.class_list_without_unknown)
    else:
        display_class_num = 5 # fix num
    return display_class_num

def visualize_grad_cam(display_class_list, display_class_num, value, model, epoch, accuracy):
    fig = plt.figure(figsize=(18, 9))
    plt.subplots_adjust(bottom=0.01)
    for main_class_index, cls in enumerate(display_class_list):
        cur_class_index = config.class_list_without_unknown.index(cls)
        img_path_list = random.sample(glob(config.val_dir + cls + '/*.jpg'), config.visualize_sample_num)
        display_image_each_class = None
        for sub_image_index, img_path in enumerate(img_path_list):
            cur_display_position = (2 * display_class_num * sub_image_index) + (2 * main_class_index) + 1
            img_path = random.choice(glob(config.val_dir + cls + '/*.jpg'))
            if config.isColor:
                img = cv2.imread(img_path, 1)
            else:
                img = cv2.imread(img_path, 0)
            if config.isColor:
                img_show = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img_show = img.copy()
            img_show = cv2.resize(img_show, (config.height, config.width))
            img = np.float32(cv2.resize(img, (config.height, config.width))) / 255
            in_tensor = preprocess_image(img).cuda()
            output = model(in_tensor)
            target_index = np.argmax(output.data.cpu().numpy())
            target_score = np.max(output.data.cpu().numpy())
            if not config.has_unknown:
                if cur_class_index == target_index :
                    is_correct = True
                else:
                    is_correct = False
            else:
                if cur_class_index == target_index and target_score > config.evaluate_threshold:
                    is_correct = True
                else:
                    is_correct = False
            if config.use_res18:
                output = output.squeeze()
            elif config.use_resnext50:
                output = output.squeeze()
            output[target_index].sum().backward(retain_graph=True)
            layer4 = value['activations']
            gradient = value['gradients']
            g = torch.mean(gradient, dim=(2,3), keepdim=True)
            grad_cam = layer4 * g
            grad_cam = torch.sum(grad_cam, dim=(0,1))
            grad_cam = torch.clamp(grad_cam, min=0)
            g_2 = gradient**2
            g_3 = gradient**3
            alpha_numer = g_2
            alpha_denom = 2 * g_2 + torch.sum(layer4 * g_3, axis=(2, 3), keepdims=True)
            alpha = alpha_numer / alpha_denom
            w = torch.sum(alpha * torch.clamp(gradient, min=0), axis=(2, 3), keepdims=True)
            grad_cam = grad_cam.data.cpu().numpy()
            grad_cam = cv2.resize(grad_cam, (config.height, config.width))

            plt.subplot(config.visualize_sample_num, 2 * display_class_num, cur_display_position)
            if cur_display_position < 2 * display_class_num and cur_display_position % 2 == 1:
                title_obj = plt.title(cls)
                plt.setp(title_obj, color='y')
            if config.isColor:
                plt.imshow(img_show)
            else:
                plt.imshow(img_show, cmap='gray')
            plt.axis('off')
            plt.subplots_adjust(wspace=0, hspace=0)
            plt.subplot(config.visualize_sample_num, 2 * display_class_num, cur_display_position + 1)
            if config.has_unknown and target_score < config.evaluate_threshold and cur_class_index == target_index:
                title_obj = plt.title("%s (%0.2f)"%(config.class_list_without_unknown[target_index], target_score))
            else:
                title_obj = plt.title(config.class_list_without_unknown[target_index])
            
            if is_correct:
                plt.setp(title_obj, color='g')
            else:
                plt.setp(title_obj, color='r')
            plt.subplots_adjust(wspace=0.1, hspace=0)
            plt.imshow(grad_cam, cmap='seismic')
            plt.imshow(img_show, alpha=.5, cmap='gray')
            plt.axis('off')
    fig.canvas.draw()
    plt.close()
    display_img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    display_img = display_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    display_img = cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR)
    cv2.putText(img=display_img, text=f"Epoch: {epoch} Accuracy: {round(accuracy, 3)}%", org=(10, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 220, 0), thickness=2, lineType=cv2.LINE_AA)
    cv2.imshow('GRAD-CAM', display_img)
    cv2.waitKey(1)

def visualize(model, epoch, accuracy):
    value = dict()
    def forward_hook(module, input, output):
        value['activations'] = output
    def backward_hook(module, input, output):
        value['gradients'] = output[0]
    target_layer = eval(config.visualize_layer)
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_backward_hook(backward_hook)
    display_class_num = get_display_class_num()
    display_class_list = random.sample(config.class_list_without_unknown, display_class_num)
    visualize_grad_cam(display_class_list, display_class_num, value, model, epoch, accuracy)

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

