import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from glob import glob
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from model import *
import shutil
from os import walk
from tqdm import tqdm
from util import *


def print_eval(class_list, real_img_count, correct_list) :
    cls_idx = 0
    for class_name in class_list :
        print("Class " , cls_idx , " : ", class_name , " Count : " , correct_list[cls_idx], " All : " , real_img_count[cls_idx] )
        if real_img_count[cls_idx] > 0 :
            print(" Accuracy : " , 100. * correct_list[cls_idx]/real_img_count[cls_idx])
        cls_idx += 1


def cf_evaluate(model_path, test_path , inpu_classes , input_width, input_height, input_ch) :

    model = Net()
    # model.load_state_dict(torch.load(path))
    model.load_state_dict(torch.load(model_path)['model_state_dict'])
    # modelsummary(model)
    class_list = ['front_real_side','horizontal','left_side','night_lights','parts','person_with_car','right_side','unknown']

    model.eval()
    # 모델의 state_dict 출력
    # print("Model's state_dict:")
    # for param_tensor in model.state_dict():
    #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    # # input_parameter = next(model.parameters())
    # # print("input_parameter : ", input_parameter , ", input_parameter.size() : ", input_parameter.size())
    # return

    no_cuda=False
    use_cuda = not no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print('use_cuda:', use_cuda, '\ndevice:', device)

    test_loss = 0
    correct = 0
    correct_list = [f * 0 for f in range(0, inpu_classes + 1)] # last index is unknown correct
    real_img_count = [f * 0 for f in range(0, inpu_classes + 1)] # last index is unknown correct

    dirs = glob(test_path + "/*")
    for dir in tqdm(dirs):
        if not os.path.isdir(dir):
            continue
        real_index = class_list.index(os.path.basename(dir))
        imgs = glob(dir + "/*.jpg")
        print("dir : " , dir, ", real_index : ", real_index, "imgs : ",  len(imgs))
        real_img_count[real_index] = len(imgs)
        for img in tqdm(imgs):
            if os.path.getsize(img) <= 0:
                os.remove(img)
                continue
            if input_ch > 1:
                image = Image.open(img)  # get image
            else:
                image = Image.open(img).convert('L')  # get image
            i = image.resize((input_width, input_height))
            # plt.imshow(i)
            # plt.show()
            trans = transforms.ToTensor()
            bi = trans(i)
            bbi = bi.unsqueeze(0)
            predict = model(bbi)
            pred_index = torch.argmax(predict).item()
            pred_confidenc = torch.max(predict).item()
            if real_index == pred_index :
                correct_list[real_index] += 1
            else :
                if pred_confidenc < 0.5:
                    correct_list[real_index] += 1
        print_eval(class_list, real_img_count, correct_list)
    print_eval(class_list, real_img_count, correct_list)


cf_evaluate(r'/home2/200_NORMAL/CF/car_cf/checkpoint/model_ep_269_loss_0.3517_accuracy_0.9571.pt',
            r'/home2/200_NORMAL/CF/car_cf/training_merge_result_org/validation',
#'/home2/200_NORMAL/CF/car_cf/training_result/validation',
            7,
            64,
            64,
            1)

