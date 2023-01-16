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
    summary(model, init_ch* config.width* config.height)
    print('=' * 20)
    print('\n')

def compose_train_transform_list(opts):
    sub_compose_list = []
    compose_list = []
    if opts['rotate'] == 1:
        sub_compose_list.append(A.Rotate(limit=(-10, +10)))
    if opts['horizontal_flip'] == 1:
        sub_compose_list.append(A.HorizontalFlip(p=1))
    if opts['rotate90'] == 1:
        sub_compose_list.append(A.RandomRotate90(p=1))
    if opts['vertical_flip'] == 1:
        sub_compose_list.append(A.VerticalFlip(p=1))
    if opts['random_brightness_contrast'] == 1:
        sub_compose_list.append(A.RandomBrightnessContrast(brightness_limit=(-0.3, 0.3), contrast_limit=(-0.3,  0.3), p=1))
    val_compose_list = compose_list.copy()
    val_compose_list.append(ToTensorV2())
    compose_list.append(A.OneOf(sub_compose_list, p=0.8))
    #compose_list.append(A.Resize(config.width, config.height))
    compose_list.append(ToTensorV2())
    return compose_list, val_compose_list

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    config = Config()
    config.summary_info()
    live_plot = LiveLossPlot()

    # no_cuda=False
    # use_cuda = not no_cuda and torch.cuda.is_available()
    # device = torch.device('cuda' if use_cuda else 'cpu')
    # model = Net().to(device)
    # modelsummary(model)

    run()
    if not os.path.exists(config.checkpoint_dir):
        os.makedirs(config.checkpoint_dir)
    torch.manual_seed(1)
    no_cuda=False
    bce_loss = nn.BCELoss()
    mse_loss = nn.MSELoss()
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
    #max_avg_accuracy = -1.
    min_avg_loss = 9999999999999
    prev_accuracy = 0.0
    if not config.knowledge_dist:
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
        model_tea = eval(config.teacher_model)()
        model_tea = model_tea.to(device)

    #modelsummary(model)
    print(model)
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
            # train_tensor = data[0].reshape(3, config.height, config.width)
            # print('train_tensor.shape:', train_tensor.shape)
            # train_img = np.array(to_pil_image(train_tensor))
            # cv2.imshow('img', train_img)
            # k = cv2.waitKey(0)
            #train_img = data.squeeze(0).reshape(config.width, config.height, 3).cpu().detach().numpy()
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
            else:
                target = target.to(device)

            output = model(data)

            loss = mse_loss(output, target).mean()
            # if config.use_ale_loss:
            #     loss = ale_loss(output, target).mean()
            # else:
            #     loss = F.binary_cross_entropy(output, target.squeeze())
                        
            optimizer.zero_grad()
            if config.knowledge_dist:
                optimizer_tea.zero_grad()
                output_tea = model_tea(data_tea)

                # if config.use_ale_loss:
                #     loss_tea = ale_loss(output_tea, target.squeeze()).mean()
                # else:
                #     loss_tea = F.binary_cross_entropy(output_tea, target.squeeze())
                loss_tea = mse_loss(output_tea, target).mean()

                output_tea = output_tea.detach()
                output_tea.requires_grad = False
                # if config.use_ale_loss:
                #     loss_dist = ale_loss(output, output_tea).mean()
                # else:
                #     loss_dist = F.binary_cross_entropy(output, output_tea)
                loss_dist = mse_loss(output, output_tea).mean()

                loss += loss_dist
                org_loss = loss - loss_dist

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
        test_loss = 0.0

        with torch.no_grad():
            for test_batch_idx, (data, target) in enumerate(test_loader):
                if config.knowledge_dist:
                    data = data['student']
                data, target = data.to(device), target.to(device)  # target: (32, class)
                output = model(data)

                input_img = data.squeeze(0).reshape(3, config.height, config.width)
                input_img = np.array(to_pil_image(input_img))
                output_img = output.squeeze(0)
                output_img = np.array(to_pil_image(output_img))

                if test_batch_idx % (config.log_interval * 2)  == 0:
                    visualize_img(input_img, output_img)

                # if config.use_ale_loss:
                #     test_loss += ale_loss(output, target.squeeze()).sum().item()
                test_loss = mse_loss(output, target).sum().item()

        total_test_len = len(test_loader.dataset)
        scheduler.step()
        if config.knowledge_dist:
            scheduler_tea.step()
        is_save = False
        if min_avg_loss > test_loss:
            min_avg_loss = test_loss
            is_save = True
            checkpoint_path = os.path.join(config.checkpoint_dir, 'model_ep_{}_loss_{:.4f}.pt'.format(epoch, test_loss))
            torch.save(model.state_dict(), checkpoint_path)

        print('='*80, '\n')

        if is_save:
            print('Best Accuracy -> save model: ', os.path.join(config.checkpoint_dir, 'model_ep_{}_loss_{:.4f}.pt'.format(epoch, test_loss)))
        print('\n')
