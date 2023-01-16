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
    
if __name__=="__main__":
    run()
    compose_directory()
    model = Net()
    print(model)
 
    path = config.predict_pretrained_model_path
    predict_unknown_threshold = config.predict_unknown_threshold
    predict_uncertain_threshold = config.predict_uncertain_threshold  #### unknown 없이 훈련했을 때 모든 클래스 스코어가 threshold보다 작으면 uncertain 폴더로 빠짐

    #model.load_state_dict(torch.load(path)['model_state_dict'])
    model.load_state_dict(torch.load(path))
    model.eval()
    
    dir = config.predict_src_path
    imgs = glob(dir + '/*.jpg')
    for img in tqdm(imgs):
        org_img = cv2.imread(img)
        org_img = cv2.resize(org_img, (config.height * 2, config.width * 2))
        if config.isColor:
            image = Image.open(img)  # get image
        else:
            image = Image.open(img).convert('L')  # get image
        i = image.resize((config.width, config.height))
        trans = transforms.ToTensor()
        bi = trans(i)
        bbi = bi.unsqueeze(0)
        #bbi = bbi.reshape(-1, config.width * config.height * 3)
    
        predict = model(bbi).squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
        predict =cv2.cvtColor(predict, cv2.COLOR_RGB2BGR)
        predict = cv2.resize(predict, (None, None), fx=2.0, fy=2.0)
        org_img = org_img / 255.0
        print('org_img.shape:', org_img.shape)
        print('predict.shape:', predict.shape)
        out = np.hstack([org_img, predict])
        cv2.imshow('rpe', out)
        k = cv2.waitKey(0)
        import sys
        if k == ord('q'):
            sys.exit(1)
        print(predict.shape)
