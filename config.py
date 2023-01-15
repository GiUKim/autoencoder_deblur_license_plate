import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import sys
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from glob import glob
from PIL import Image
import os
import numpy as np

class Config:
    def __init__(self):
        self.info = self.read_config()
        # hyperparameter tuning
        self.train_dir = self.info['TRAIN_PATH']
        self.val_dir = self.info['VAL_PATH']
        ckpt_dir = self.info['CHECKPOINT_PATH']
        self.pretrained_model = self.info['PRETRAINED_MODEL_PATH']
        self.init_best_accuracy_for_pretrained_model = self.info[
         'INIT_BEST_ACCURACY_FOR_PRETRAINED_MODEL']  #### best accuracy 0으로 초기화 할 것인지? -> validation 데이터 추가되면 True, 그대로면 False
        self.min_lr = self.info['MIN_LEARNING_RATE']
        self.max_lr = self.info['MAX_LEARNING_RATE']
        self.momentum = self.info['MOMENTUM']
        self.batch_size = self.info['BATCH_SIZE']
        self.test_batch_size = self.info['TEST_BATCH_SIZE']
        self.width = self.info['WIDTH']
        self.height = self.info['HEIGHT']
        self.evaluate_threshold = self.info['EVALUATE_UNKNOWN_THRESHOLD']
        self.label_smoothing = self.info['LABEL_SMOOTHING']
        self.label_smoothing_scale = self.info['LABEL_SMOOTHING_SCALE']
        self.augmentation_options = {
                    'center_crop': self.info['CENTER_CROP'],                        # 이미지 한가운데 80% 영역만 크롭
                    'horizontal_flip': self.info['HORIZONTAL_FLIP'],                    # 좌우 반전
                    'rotate90': self.info['ROTATE_90'],                           # 90도 기울기
                    'rotate': self.info['ROTATE'],                             # 랜덤 기울기 
                    'vertical_flip': self.info['VERTICAL_FLIP'],                      # 상하 반전 
                    'optical_distortion': self.info['OPTICAL_DISTORTION'],                 # 렌즈 왜곡 효과
                    'random_brightness_contrast': self.info['RANDOM_BRIGHTNESS_CONTRAST'],         # 밝기 랜덤 조절 (밝게, 어둡게) 
                    'channel_shuffle': self.info['CHANNEL_SHUFFLE'],                    # 채널 간 반전 ex) RGB->BGR 등으로 변경
                    'cutout': self.info['CUTOUT'],                             # 이미지에 랜덤하게 black hole 생성
                    'custom_lattepyo': self.info['CUSTOM_LATTEPYO']                     # custom aug
                }
        self.smoothing_scale = self.info['LABEL_SMOOTHING_SCALE']
        self.ale_gamma = self.info['ALE_LOSS_FOCAL_GAMMA_SCALE']
        self.epochs = self.info['EPOCHS']
        self.log_interval = self.info['LOG_INTERVAL']
        self.dim = self.info['DIM']
        self.isColor = self.info['IS_COLOR']
        self.DEBUG_MODE = self.info['DEBUG_MODE']
        self.use_custom_lr = self.info['USE_CUSTOM_LR']
        self.use_ale_loss = self.info['USE_ALE_LOSS']

        # use resnet18 layer (10M parameter)
        self.use_res18 = self.info['USE_RESNET18']
        self.use_resnext50 = self.info['USE_RESNEXT50']

        self.knowledge_dist = self.info['KNOWLEDGE_DISTILLATION']
        
        # grad_cam
        self.visualize_grad_cam = self.info['VISUALIZE_GRAD_CAM']
        self.visualize_sample_num = self.info['VISUALIZE_SAMPLE_NUM']
        self.visualize_period = self.info['VISUALIZE_PERIOD'] # N * (batch_size) iterator 마다
#        self.visualize_layer = 'model.cbr9'
        self.visualize_layer = self.info['VISUALIZE_LAYER']
        
        # predict model parameter
        self.predict_src_path = self.info['PREDICT_SOURCE_PATH']
        self.predict_dst_path = self.info['PREDICT_DESTINATION_PATH']
        self.predict_confidence_divide = self.info[
         'PREDICT_CONFIDENCE_DIVIDE_RULE']  # edit here -> ex) OVER 0, 10, 30, 60, 90
        self.predict_pretrained_model_path = self.info['PREDICT_PRETRAINED_MODEL_PATH']
        self.predict_remove_exist_dir = self.info['PREDICT_REMOVE_DIRECTORY_TREE']
        self.predict_verbose_score = self.info['PREDICT_VERBOSE_SCORE']  # 저장된 이미지 이름에 스코어 표시?
        
        self.predict_unknown_threshold = self.info['PREDICT_UNKNOWN_THRESHOLD']   # unknown 결정 threshold
        self.predict_uncertain_threshold = self.info['PREDICT_UNCERTAIN_THRESHOLD']  # unknown없이 학습한 경우 모든 클래스 스코어가 이거보다 낮으면 uncertain 예측으로 따로 걸러냄        


        # 수정 X
        if self.train_dir[-1] != '/':
            self.train_dir += '/'
        if self.val_dir[-1] != '/':
            self.val_dir += '/'
        self.checkpoint_dir = ckpt_dir
        self.train_paths = glob(self.train_dir + '*/*.jpg')
        self.test_paths = glob(self.val_dir + '*/*.jpg')
        self.predict_confidence_divide = [0.0] + self.predict_confidence_divide + [1.0 + 1e-5]

        self.class_list = sorted([f.split('/')[-1] for f in glob(self.train_dir + '*')])
        self.class_list_without_unknown = sorted(list(set(self.class_list) - set(['unknown'])))
        self.has_unknown = 'unknown' in self.class_list
        self.unknown_idx, self.num_classes = self.get_unknown_idx_num_classes()
        self.test_class_each_num = self.get_test_class_each_num()
        if self.has_unknown:
            self.test_unknown_len = len(glob(self.val_dir + 'unknown/*.jpg'))

        if (self.use_res18 or self.use_resnext50) and not self.knowledge_dist:
            if self.use_res18:
                self.visualize_layer = 'model.layer4'
            elif self.use_resnext50:
                self.visualize_layer = 'model.layer4'
            self.width = 224
            self.height = 224
            self.isColor = True

        if self.knowledge_dist:
            self.visualize_layer = self.info['VISUALIZE_LAYER']
            self.width_tea = self.info['TEACHER_WIDTH']
            self.height_tea = self.info['TEACHER_HEIGHT']
            self.isColor_tea = self.info['TEACHER_IS_COLOR']
            self.teacher_model = self.info['TEACHER_MODEL']

    def get_unknown_idx_num_classes(self):
        if self.has_unknown:
            return self.class_list.index('unknown'), len(self.class_list) - 1
        else:
            return -1, len(self.class_list)

    def get_test_class_each_num(self):
        test_class_each_num = []
        for cls in self.class_list:
            cls_obj_num = len(glob(self.val_dir + cls + '/*.jpg'))
            test_class_each_num.append(cls_obj_num)
        return test_class_each_num
    def read_config(self):
        try:
            f = open('.config', 'r')    
        except:
            print(f"File not exsits [.config] error, config file must be named [.config]")
            sys.exit(1)
        ls = f.readlines()
        f.close()
        info = {}
        for l in ls:
            if l == '\n' or l == '' or '#' in l or '---' in l:
                continue
            datas = l.split(' ')
            NAME = datas[0]
            DATA = (l.split(NAME)[-1].split('\n')[0]).lstrip()
            info[NAME] = eval(DATA)
        return info
    def summary_info(self):
        print('-'*100)
        print('CONFIG INFO')
        for k, v in self.info.items():
            strFormat = '%-50s%-50s'
            strOut = strFormat % (k, v)
            print(strOut)

        print('-'*100)

