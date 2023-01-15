from config import Config
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from glob import glob
from PIL import Image
from config import Config
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from glob import glob
from PIL import Image
from config import Config
from config import *
import cv2
# config = Config(tuttrain.train_dir, tuttrain.test_dir, tuttrain.checkpoint_dir)

config = Config()
def get_label(path):
    lbl_name = path.split('/')[-2]
    # lbl_name = path.split('\\')[-2]
    lbl_idx = config.class_list.index(lbl_name)
    if config.has_unknown:
        if lbl_idx < config.unknown_idx:
            label = np.eye(len(config.class_list) - 1, dtype=np.float32)[lbl_idx]
        elif lbl_idx == config.unknown_idx:
            label = np.zeros((len(config.class_list) - 1), dtype=np.float32)
        else:
            label = np.eye(len(config.class_list) - 1, dtype=np.float32)[lbl_idx - 1]
        if config.label_smoothing:
            label[np.where(label < 0.5)] += config.label_smoothing_scale
            label[np.where(label > 0.5)] -= config.label_smoothing_scale
    else:
        label = np.eye(len(config.class_list), dtype=np.float32)[lbl_idx]
        if config.label_smoothing:
            label[np.where(label < 0.5)] += config.label_smoothing_scale
            label[np.where(label > 0.5)] -= config.label_smoothing_scale
    return label

class Dataset(Dataset):
    def __init__(self, data_paths, transform=None):
        self.data_paths = data_paths
        self.transform = transform

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        path = self.data_paths[idx]
        if config.isColor:
            image = Image.open(path) # get image
        else:
            image = Image.open(path).convert('L') # get image
        if config.knowledge_dist and config.isColor_tea:
            image_tea = Image.open(path)
        elif config.knowledge_dist and not config.isColor_tea:
            image_tea = Image.open(path).convert('L')
        image = image.resize((config.width, config.height))
        image = np.array(image, dtype=np.float32)
        image = image / 255.
        if config.isColor:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = get_label(path) # get label
        if self.transform:
            image = self.transform(image=image)["image"]
        label = torch.Tensor(label)
        if not config.knowledge_dist:
            return image, label
        else:
            image_tea = image_tea.resize((config.width_tea, config.height_tea))
            image_tea = np.array(image_tea, dtype=np.float32)
            image_tea = image_tea / 255.
            if config.isColor_tea:
                image_tea = cv2.cvtColor(image_tea, cv2.COLOR_BGR2RGB)
            if self.transform:
                image_tea = self.transform(image=image_tea)["image"]

            return {'student': image, 'teacher': image_tea}, label

