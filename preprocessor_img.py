from config import Config
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from glob import glob
from config import Config
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from config import *
import cv2
import os
import random
import shutil
from tqdm import tqdm
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import matplotlib.pyplot as plt

config = Config()


def transform_img_to_resize(img, img_width, img_height):
    # Histogram Equalization
    # img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    # img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    # img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

    # Image Resizing
    height, width = img.shape[:2]

    if width > img_width or height > img_height:
        img = cv2.resize(img, (img_width, img_height), cv2.INTER_LINEAR)
    elif width < img_width or height < img_height:
        img = cv2.resize(img, (img_width, img_height), cv2.INTER_LINEAR)

    return img

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
    # Histogram Equalization
    # img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    # img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    # img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

    # Image Resizing
    height, width = img.shape[:2]

    if width > img_width or height > img_height:
        img = cv2.resize(img, (img_width, img_height), cv2.INTER_LINEAR)
    elif width < img_width or height < img_height:
        img = cv2.resize(img, (img_width, img_height), cv2.INTER_LINEAR)

    return img

def change_feature_image(img_path) :

    img = cv2.imread(img_path)
    img_denose = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    if img.shape[2] != 1 :
        gray = cv2.cvtColor(img_denose, cv2.COLOR_BGR2GRAY)
    else :
        gray = img_denose

    # height, width = img.shape[:2]
    # print("height , width : ", height, width)
    # if config.width != width or config.height != height:
    #     gray = transform_img_to_resize(gray, config.width, config.height)

    kernel = get_gaussian_filter()
    gray = cv2.filter2D(gray, -1, kernel)
    kernel = get_laplacian_black_filter()
    binary1 = cv2.filter2D(gray, -1, kernel)
    t, t_otsu = cv2.threshold(binary1, -1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # contours , hierachy = cv2.findContours(t_otsu, cv2.RETR_TREE , cv2.CHAIN_APPROX_TC89_KCOS)
    contours, hierachy = cv2.findContours(t_otsu, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)
    img_result = gray.copy().astype('uint8')
    cv2.drawContours(img_result, contours, -1, (0, 0, 0), 1)
    # cv2.drawContours(img_result, contours, -1, (255, 255, 255), 1)


    # titles = ['img', 'img_denose', 't_otsu', 'img_result']
    # images = [img, img_denose, t_otsu, img_result]
    # for i in range(len(images)):
    #     plt.subplot(1, 4, i + 1)
    #     plt.title(titles[i])
    #     plt.imshow(images[i], cmap='gray')
    #     plt.axis('off')
    # plt.tight_layout()
    # plt.show()
    # cv2.imshow("img_result" , img_result)
    # cv2.waitKey()
    return img_result , t_otsu

class_list = ['front_real_side', 'horizontal' , 'left_side' , 'night_lights', 'parts' , 'person_with_car', 'right_side' , 'unknown']
g_origin_path = "data_set_org3"
g_result_path = "data_set_temp_pre_result"
g_result_bin_path = "data_set_temp_pre_bin"

def convert_save_train_data(train_data_path) :
    global g_origin_path, g_result_path, g_result_bin_path

    src_path = os.path.join(train_data_path,g_origin_path)
    result_path = os.path.join(train_data_path,g_result_path)
    if not os.path.exists(result_path) :
        os.mkdir(result_path)

    for cls in class_list :
        src_cls_path = os.path.join(src_path,cls)
        result_cls_path = os.path.join(result_path, cls)

        if os.path.isdir(src_cls_path) :
            print("src_cls_path : ", src_cls_path)
            if not os.path.isdir(result_cls_path) :
                os.mkdir(result_cls_path)
            imgs = glob(src_cls_path + "/*.jpg")
            for img in tqdm(imgs) :
                # if os.path.exists (os.path.join(result_cls_path,os.path.basename(img))) :
                #     continue
                print("img : ", img)
                result_img, result_bin_img = change_feature_image(img)
                cv2.imwrite(os.path.join(result_cls_path,os.path.basename(img)), result_img)
                # break


convert_save_train_data(r"/home2/200_NORMAL/CF/car_cf")



# def random_file_copy(src_path, dst_path, max_count):
#     imgs = glob(src_path + "\*.jpg")
#     random.shuffle(imgs)
#     print(len(imgs))
#
#     count = 0
#     for img in imgs :
#         count += 1
#         print("count:", count , ", " , img , " -> " , dst_path)
#         shutil.move(img,dst_path)
#         if count >= max_count :
#             break

# random_file_copy(r"J:\200_NORMAL\OTHER_CROP\car_unknown", r"T:\200_NORMAL\CF\car_cf\origin_images\unknown", 5000)
