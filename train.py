import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, CosineAnnealingWarmRestarts
from glob import glob
from PIL import Image
from config import Config
from dataset import *
from model import *
import random
import sys
import os
import time
from time import process_time
import math
import numpy as np
from tqdm import tqdm
from torchsummary import summary as summary
from util import *
from torch.utils.tensorboard import SummaryWriter
from live_loss_plot import *
from torchvision.transforms.functional import to_pil_image
import torchvision
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.models import resnet18, resnext50_32x4d 
import warnings

def run():
    torch.multiprocessing.freeze_support()
    print('loop')

def modelsummary(model):
    print('=' * 20)
    if config.isColor:
        init_ch = 3
    else:
        init_ch = 1
    summary(model, (init_ch, config.width, config.height))
    print('=' * 20)
    print('\n')

def compose_train_transform_list(opts):
    sub_compose_list = []
    compose_list = []
    if opts['center_crop'] == 1:
        sub_compose_list.append(A.CenterCrop(config.width * 8 // 10, config.height * 8 // 10))
    if opts['rotate'] == 1:
        sub_compose_list.append(A.Rotate(limit=(-10, +10)))
    if opts['custom_lattepyo'] == 1:
        sub_compose_list.append(A.Lambda(name='Lambda', image=apply_custom_aug, p=1))
    if opts['horizontal_flip'] == 1:
        sub_compose_list.append(A.HorizontalFlip(p=1))
    if opts['rotate90'] == 1:
        sub_compose_list.append(A.RandomRotate90(p=1))
    if opts['vertical_flip'] == 1:
        sub_compose_list.append(A.VerticalFlip(p=1))
    if opts['optical_distortion'] == 1:
        sub_compose_list.append(A.OpticalDistortion(p=1))
    if opts['random_brightness_contrast'] == 1:
        sub_compose_list.append(A.RandomBrightnessContrast(brightness_limit=(-0.3, 0.3), contrast_limit=(-0.3,  0.3), p=1))
    if opts['channel_shuffle'] == 1 and config.isColor:
        sub_compose_list.append(A.ChannelShuffle(p=1))
    if opts['cutout'] == 1:
        sub_compose_list.append(A.Cutout(p=1, num_holes=8, max_h_size=8, max_w_size=8))
    val_compose_list = compose_list.copy()
    val_compose_list.append(ToTensorV2())
    compose_list.append(A.OneOf(sub_compose_list, p=0.8))
    compose_list.append(A.Resize(config.width, config.height))
    compose_list.append(ToTensorV2())
    return compose_list, val_compose_list

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    config = Config()
    config.summary_info()
    live_plot = LiveLossPlot()
    run()
    if not os.path.exists(config.checkpoint_dir):
        os.makedirs(config.checkpoint_dir)
    torch.manual_seed(1)
    no_cuda=False
    bce_loss = nn.BCELoss()
    ale_loss = ALE_loss(gamma=config.ale_gamma)
    use_cuda = not no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print('use_cuda:', use_cuda, '\ndevice:', device)
    os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2, 3'
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    kwargs = {'num_workers': 12, 'pin_memory': True} if use_cuda else {}
    train_compose_list, val_compose_list = compose_train_transform_list(config.augmentation_options)
    train_transforms = A.Compose(train_compose_list)
    val_transforms = A.Compose(val_compose_list)
    train_loader = torch.utils.data.DataLoader(
        Dataset(config.train_paths,
                train_transforms),
        batch_size=config.batch_size,
        shuffle=True,
        **kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        Dataset(config.test_paths,
                val_transforms),
        batch_size=config.test_batch_size,
        shuffle=False,
        **kwargs
    )

    start_epoch = 1
    max_avg_accuracy = -1.
    prev_accuracy = 0.0
    if not config.knowledge_dist:
        if config.use_res18:
            model = call_resnet18()
            model = model.to(device)
        elif config.use_resnext50:
            model = call_resnext50_32x4d()
            model = model.to(device)
        else:
            model = Net().to(device)
        print("config.pretrained_model : ", config.pretrained_model)
        if config.pretrained_model is not None and os.path.exists(config.pretrained_model):
            model_info = torch.load(config.pretrained_model)
            model.load_state_dict(model_info['model_state_dict'])
            start_epoch = model_info['epoch'] + 1
            print(f'[Select pretrained-model: {config.pretrained_model}]')
            print(f'[Train epoch: from {start_epoch} to {config.epochs+1}]\n')
            if config.init_best_accuracy_for_pretrained_model:
                max_avg_accuracy = float(config.pretrained_model.split('/')[-1].split('accuracy_')[-1].split('.pt')[0]) * 100.
                prev_accuracy = max_avg_accuracy
                print(f'[pretrained-model best accuracy: {round(max_avg_accuracy, 3)}]')

    else:
        model = Net().to(device)
        if config.teacher_model == 'resnext50_32x4d':
            model_tea = call_resnext50_32x4d()
        elif config.teacher_model == 'resnet18':
            model_tea = call_resnet18()
        else:
            model_tea = eval(config.teacher_model)()
        model_tea = model_tea.to(device)

    modelsummary(model)
    print(model)
#    model = MLP_Mixer((1, config.width, config.height), 16, 64, 32, config.num_classes).to(device)
    if config.use_custom_lr:
        optimizer = optim.Adam(model.parameters(), lr=config.max_lr)
        scheduler = LRScheduler(optimizer, warm_up=0.5, total_epoch=config.epochs)
    else:
        optimizer = optim.SGD(model.parameters(), lr=config.max_lr, momentum=config.momentum)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, eta_min=config.min_lr, verbose=True)
    if config.knowledge_dist:
        if config.use_custom_lr:
            optimizer_tea = optim.Adam(model_tea.parameters(), lr=config.max_lr)
            scheduler_tea = LRScheduler(optimizer_tea, warm_up=0.5, total_epoch=config.epochs)
        else:
            optimizer_tea = optim.SGD(model_tea.parameters(), lr=config.max_lr, momentum=config.momentum)
            scheduler_tea = CosineAnnealingWarmRestarts(optimizer_tea, T_0=10, eta_min=config.min_lr, verbose=False)

#    scheduler = CosineAnnealingLR(optimizer, T_max=20, eta_min=config.min_lr, verbose=True)
    for epoch in range(start_epoch, config.epochs+1):
        model.train()
        if config.knowledge_dist:
            model_tea.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            if config.knowledge_dist:
                data_tea = data['teacher']
                data = data['student']
            if config.DEBUG_MODE:
                for i in range(4):
                    img = np.array(to_pil_image(data[i]))
                    cv2.imshow('image', img)
                    k = cv2.waitKey(0)
                    if k == ord('q'):
                        sys.exit(1)
             
            if batch_idx % config.visualize_period == 0 and config.visualize_grad_cam:
                model.eval()
                visualize(model, epoch, prev_accuracy)
                model.train()

            if device is not None:
                data = data.cuda(device, non_blocking=True)
                if config.knowledge_dist:
                    data_tea = data_tea.cuda(device, non_blocking=True)
            else :
                data = data.to(device)
                if config.knowledge_dist:
                    data_tea = data_tea.to(device)

            if torch.cuda.is_available():
                target = target.cuda(device, non_blocking=True)
            else :
                target = target.to(device)

            output = model(data)
            output = output.squeeze(0)
            if config.use_ale_loss:
                loss = ale_loss(output, target.squeeze()).mean()
            else:
                loss = F.binary_cross_entropy(output, target.squeeze())
                        
            optimizer.zero_grad()
            if config.knowledge_dist:
                optimizer_tea.zero_grad()
                output_tea = model_tea(data_tea)
                #loss_tea = F.binary_cross_entropy(output_tea, target.squeeze())
                if config.use_ale_loss:
                    loss_tea = ale_loss(output_tea, target.squeeze()).mean()
                else:
                    loss_tea = F.binary_cross_entropy(output_tea, target.squeeze())
                    #loss_tea = bce_loss(output_tea, target.squeeze())
                output_tea = output_tea.detach()
                output_tea.requires_grad = False
                if config.use_ale_loss:
                    loss_dist = ale_loss(output, output_tea).mean()
                else:
                    loss_dist = F.binary_cross_entropy(output, output_tea)

                #loss_dist = bce_loss(output, output_tea)
                loss += loss_dist
                org_loss = loss - loss_dist
    
            #if config.has_unknown and len(config.class_list) == 2:
            #    output = output.view(len(output), 1)
            live_plot.update(loss=loss.item())
            loss.backward()
            if config.knowledge_dist:
                loss_tea.backward()
                optimizer_tea.step()
            optimizer.step()
            if config.knowledge_dist:
                print('Train Epoch: {:>2} [{:>6}/{:>6} ({:.0f}%)] {:<4}: {:.6f} / {:.6f}'.format(
                    epoch, batch_idx*len(data), len(train_loader.dataset), 100.*batch_idx/len(train_loader), "Loss(Student / Teacher)", org_loss.item(), loss_tea.item()
                ), end='\r')
            if batch_idx % config.log_interval == 0:
                print('Train Epoch: {:>2} [{:>6}/{:>6} ({:.0f}%)] {:<4}: {:.6f}'.format(
                    epoch, batch_idx*len(data), len(train_loader.dataset), 100.*batch_idx/len(train_loader), "Loss", loss.item()
                ), end='\r')
        model.eval()
        test_loss = 0
        correct = 0
        correct_list = [f * 0 for f in range(0, config.num_classes + 1)] # last index is unknown correct
        
        with torch.no_grad():
            for data, target in test_loader:
                if config.knowledge_dist:
                    data = data['student']
                data, target = data.to(device), target.to(device)  # target: (32, class)
                output = model(data)
                output = output.squeeze(0)
                #if config.has_unknown and len(config.class_list) == 2:
                #    output = output.view(len(output), 1)
                if config.use_ale_loss:
                    test_loss += ale_loss(output, target.squeeze()).sum().item()
                test_loss += F.binary_cross_entropy(output, target.squeeze(), reduction='sum').item()
                for idx in range(0, len(target)):
                    pred = output[idx]
                    pred_max_value = torch.max(pred)
                    if pred_max_value.item() < config.evaluate_threshold and config.has_unknown:  # pred to unknown
                        if torch.sum(target[idx]).item() < 0.5:  # unknown correct
                            #correct += 1  # unknown은 전체 통계 제외
                            correct_list[-1] += 1
                    else:
                        if torch.argmax(target[idx]) == torch.argmax(pred) and torch.max(target[idx]).item() > 0.5:
                            correct += 1
                            correct_list[torch.argmax(target[idx])] += 1

        if config.has_unknown:
            total_test_len = len(test_loader.dataset) - config.test_unknown_len # unknown 통계 제외
        else:
            total_test_len = len(test_loader.dataset)
        test_loss /= total_test_len
        accuracy = 100.*correct / total_test_len
            
        prev_accuracy = accuracy
        scheduler.step()
        if config.knowledge_dist:
            scheduler_tea.step()
        is_save = False
        if max_avg_accuracy < accuracy: # svz model pth
            max_avg_accuracy = accuracy
            is_save = True
            checkpoint_path = os.path.join(config.checkpoint_dir, 'model_ep_{}_loss_{:.4f}_accuracy_{:.4f}.pt'.format(epoch, test_loss, accuracy/100.))
            torch.save(model.state_dict(), checkpoint_path)
#            torch.save({'model_state_dict':model.state_dict(),
 #                      'epoch': epoch
   #                     },
  #                    checkpoint_path)

        print('='*80, '\n')
        print('Epochs: {}, Test set: Average Loss: {:4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
                epoch, test_loss, correct, total_test_len, accuracy
            )
        )
        print('=' * 80)

        print_validation_eval_log(correct_list)

        if is_save:
            print('Best Accuracy -> save model: ', os.path.join(config.checkpoint_dir, 'model_ep_{}_loss_{:.4f}_accuracy_{:.4f}.pt'.format(epoch, test_loss, accuracy/100.)))
        print('\n')
