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
import shutil
import matplotlib.pyplot as plt
from model import *
from tqdm import tqdm
from torchvision.models import resnet18

def run():
    torch.multiprocessing.freeze_support()
    print('loop')
no_cuda=False
use_cuda = not no_cuda and torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2, 3'
print('device:', device)
print('cuda:', use_cuda)

config = Config()

def compose_directory():
    if not os.path.exists(config.predict_dst_path):
        os.makedirs(config.predict_dst_path)
    else:
        if config.predict_remove_exist_dir:
            shutil.rmtree(config.predict_dst_path)
            os.makedirs(config.predict_dst_path)
    
    divide_dir_list = []
    if len(config.predict_confidence_divide) > 2:
        for idx, c in enumerate(config.predict_confidence_divide):
            if idx == len(config.predict_confidence_divide) - 1:
                break
            divide_dir_list.append(f"OVER_{int(c * 100)}")
    print('len(devide_dir_list):', len(divide_dir_list))
    if len(divide_dir_list) == 0:
        for cls in config.class_list:
            if not os.path.exists(os.path.join(config.predict_dst_path, cls)):
                os.makedirs(os.path.join(config.predict_dst_path, cls))
            else:
                if config.predict_remove_exist_dir:
                    shutil.rmtree(os.path.join(config.predict_dst_path, cls))
                    os.makedirs(os.path.join(config.predict_dst_path, cls))
        if not config.has_unknown:
            if not os.path.exists(os.path.join(config.predict_dst_path, 'uncertain')):
                os.makedirs(os.path.join(config.predict_dst_path, 'uncertain'))
            else:
                if config.predict_remove_exist_dir:
                    shutil.rmtree(os.path.join(config.predict_dst_path, 'uncertain'))
                    os.makedirs(os.path.join(config.predict_dst_path, 'uncertain'))
    else:
        for cls in config.class_list:
            if cls == 'unknown':
                if not os.path.exists(os.path.join(config.predict_dst_path, 'unknown')):
                    os.makedirs(os.path.join(config.predict_dst_path, 'unknown'))
                else:
                    if config.predict_remove_exist_dir:
                        shutil.rmtree(os.path.join(config.predict_dst_path, 'unknown'))
                        os.makedirs(os.path.join(config.predict_dst_path, 'unknown'))
                continue
    
            for divide_dir in divide_dir_list:
                if not os.path.exists(os.path.join(config.predict_dst_path, cls, divide_dir)):
                    os.makedirs(os.path.join(config.predict_dst_path, cls, divide_dir))
                else:
                    if config.predict_remove_exist_dir:
                        shutil.rmtree(os.path.join(config.predict_dst_path, cls, divide_dir))
                        os.makedirs(os.path.join(config.predict_dst_path, cls, divide_dir))
        if not config.has_unknown:
            if not os.path.exists(os.path.join(config.predict_dst_path, 'uncertain')):
                os.makedirs(os.path.join(config.predict_dst_path, 'uncertain'))

if __name__=="__main__":
    run()
    compose_directory()
    model = Net()
    ############
    # ----- resnet 18 use -> size(224, 224, isColor=True 자동 수정), grad-cam(cbr9->layer4 로 쟈동 수정됨) #
    if config.use_res18:
        model = call_resnet18()
    elif config.use_resnext50:
        model = call_resnext50_32x4d()
    ###########
    print(model)
 
    path = config.predict_pretrained_model_path
    verbose_score = config.predict_verbose_score ######## 파일명 뒤에 예측스코어값 붙여서 저장할건지?
    predict_unknown_threshold = config.predict_unknown_threshold
    predict_uncertain_threshold = config.predict_uncertain_threshold  #### unknown 없이 훈련했을 때 모든 클래스 스코어가 threshold보다 작으면 uncertain 폴더로 빠짐

    model.load_state_dict(torch.load(path)['model_state_dict'])
    model.eval()
    
    dir = config.predict_src_path
    imgs = glob(dir + '/*.jpg')
    for img in tqdm(imgs):
        if config.isColor:
            image = Image.open(img)  # get image
        else:
            image = Image.open(img).convert('L')  # get image
        i = image.resize((config.width, config.height))
        trans = transforms.ToTensor()
        bi = trans(i)
        bbi = bi.unsqueeze(0)
    
        predict = model(bbi)
        if config.use_res18:
            predict = predict.squeeze()
        elif config.use_resnext50:
            predict = predict.squeeze()
        print('--------------------------')
        print(img.split('/')[-1])
        print('score ->', [round(f, 3) for f in predict.tolist()])
        print('--------------------------')
        
        if config.has_unknown:
            if torch.max(predict) < predict_unknown_threshold:
                maxindex = torch.argmax(predict)
                if verbose_score:
                    score = str(round(predict[maxindex].item(), 4))
                    shutil.copy(img, os.path.join(config.predict_dst_path + '/unknown', img.split('/')[-1].split('.jpg')[0]+'_['+score+'].jpg'))
                else:
                    shutil.copy(img, os.path.join(config.predict_dst_path + '/unknown', img.split('/')[-1]))
            else:
                maxindex = torch.argmax(predict)
                print('predict[maxindex]:', predict[maxindex])
                score = str(round(predict[maxindex].item(), 4))
                if len(config.predict_confidence_divide) > 2:
                    for idx, c in enumerate(config.predict_confidence_divide):
                        if float(c) > float(score):
                            area_name = f"OVER_{int(config.predict_confidence_divide[idx - 1] * 100)}"
                            break
                        else:
                            area_name = ''
                else:
                    area_name = ''
                if verbose_score:
                    if maxindex < config.class_list.index('unknown'):
                        shutil.copy(img, os.path.join(config.predict_dst_path, config.class_list[maxindex], area_name, img.split('/')[-1].split('.jpg')[0]+'_['+score+'].jpg'))
                    else:
                        shutil.copy(img, os.path.join(config.predict_dst_path, config.class_list[maxindex + 1], area_name, img.split('/')[-1].split('.jpg')[0]+'_['+score+'].jpg'))
                else:
                    if maxindex < config.class_list.index('unknown'):
                        shutil.copy(img, os.path.join(config.predict_dst_path, config.class_list[maxindex], area_name))
                    else:
                        shutil.copy(img, os.path.join(config.predict_dst_path, config.class_list[maxindex + 1], area_name))
        else:
            maxindex = torch.argmax(predict) 
            score = str(round(predict[maxindex].item(), 4))
            if len(config.predict_confidence_divide) > 2:
                for idx, c in enumerate(config.predict_confidence_divide):
                    if float(c) > float(score):
                        area_name = f"OVER_{int(config.predict_confidence_divide[idx - 1] * 100)}"
                        break
                    else:
                        area_name = ''
            else:
                area_name = ''

            if predict[maxindex].item() > predict_uncertain_threshold:
                if verbose_score:
                    shutil.copy(img, os.path.join(config.predict_dst_path, config.class_list[maxindex], area_name, img.split('/')[-1].split('.jpg')[0]+'_['+score+'].jpg'))
                else:
                    shutil.copy(img, os.path.join(config.predict_dst_path, config.class_list[maxindex], area_name))
            else:
                if verbose_score:
                    shutil.copy(img, os.path.join(config.predict_dst_path + '/uncertain', img.split('/')[-1].split('.jpg')[0]+'_['+score+'].jpg'))
                else:
                    shutil.copy(img, os.path.join(config.predict_dst_path, 'uncertain'))
