# Build training pipeline for all beseline methods
# Date: 06/28/2020
# Author: Jin Huang. Based on Dawei and X3D official code

import os
import argparse

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pkbar
import pdb
import numpy as np
import x3d as resnet_x3d

from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchsummary import summary
from torchvision import datasets, transforms

from barbar import Bar
from utils.apmeter import APMeter
from data.ucf101 import customized_dataset
from transforms.spatial_transforms import Compose, Normalize, \
    RandomHorizontalFlip, MultiScaleRandomCrop, \
    MultiScaleRandomCropMultigrid, ToTensor, \
    CenterCrop, CenterCropScaled
from transforms.temporal_transforms import TemporalRandomCrop
from transforms.target_transforms import ClassLabel

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('-gpu', default='0', type=str)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

# TODO: Build baseline pipeline for both HMDB51 and UCF101

####################################################
# Setup paths and parameters
####################################################
dataset = "hmdb_ta2" # choose among ["ucf101", "ucf101_ta2", "hmdb", "hmdb_ta2"]
use_pretrain = True # Pretrain model is from Kinetics 400

if use_pretrain:
    pretrain_model_path = "models/x3d_multigrid_kinetics_fb_pretrained.pt"

# UCF101 official partition
if dataset == "ucf101":
    nb_classes = 101

    npy_root_dir = ""
    anno_json_path = ""
    model_save_dir = ""


# UCF101 SAIL-ON partition
elif dataset == "ucf101_ta2":
    nb_classes = 88

    npy_root_dir = ""
    anno_json_path = "/data/jin.huang/ucf101_npy_json/ucf101.json"
    model_save_dir = ""


# HMDB51 official partition
elif dataset == "hmdb51":
    nb_classes = 51

    npy_root_dir = ""
    anno_json_path = ""
    model_save_dir = ""


# HMDB51 SAIL-ON partition
elif dataset == "hmdb_ta2":
    nb_classes = 26

    npy_root_dir = "/data/jin.huang/hmdb51/npy_json/0"
    anno_json_path = "/data/jin.huang/hmdb51/npy_json/0/ta2_partition_0.json"
    model_save_dir = "/data/jin.huang/hmdb51_models/0614_x3d_p0"

    data_mean = [0, 0, 0]
    data_std = [1, 1, 1]

####################################################
# Usually, no need to change these
####################################################
nb_gpu = 1
batch = 16
nb_epoch = 100
batch_upscale = 2
nb_workers = 12
init_lr = 0.02 * batch_upscale

X3D_VERSION = 'M'
use_long_cycle = False
cont_training = False

if use_long_cycle:
    pass

if cont_training:
    previous_model_path = None


frames=80 # DOUBLED INSIDE DATASET, AS LONGER CLIPS
crop_size = {'S':160, 'M':224, 'XL':312}[X3D_VERSION]
gamma_tau = {'S':6, 'M':5, 'XL':5}[X3D_VERSION]

if not use_long_cycle:
    resize_size = {'S': [180., 225.],
                   'M': [256., 256.],
                   'XL': [360., 450.]}[X3D_VERSION]
else:
    resize_size = {'S':[180.,225.],
                   'M':[256.,320.],
                   'XL':[360.,450.]}[X3D_VERSION]

TA2_DATASET_SIZE = {'train':13446, 'val':1491}
TA2_MEAN = [0, 0, 0]
TA2_STD = [1, 1, 1]


####################################################
# Usually, no need to change these
####################################################
def lr_warmup(init_lr,
              cur_steps,
              warmup_steps,
              opt):
    """

    """
    start_after = 1
    if cur_steps < warmup_steps and cur_steps > start_after:
        lr_scale = min(1., float(cur_steps + 1) / warmup_steps)
        for pg in opt.param_groups:
            pg['lr'] = lr_scale * init_lr


def print_stats(long_ind,
                batch_size,
                stats,
                gamma_tau,
                bn_splits,
                lr):
    """

    """
    bs = batch_size * LONG_CYCLE[long_ind]
    if long_ind in [0,1]:
        bs = [bs*j for j in [2,1]]
        print(' ***** LR {} '
              'Frames {}/{} '
              'BS ({},{}) '
              'W/H ({},{}) '
              'BN_splits {} '
              'long_ind {} *****'.format(lr,
                                         stats[0][0], gamma_tau,
                                         bs[0], bs[1],
                                         stats[2][0], stats[3][0],
                                         bn_splits,
                                         long_ind))
    else:
        bs = [bs*j for j in [4,2,1]]
        print(' ***** LR {} '
              'Frames {}/{} '
              'BS ({},{},{}) '
              'W/H ({},{},{}) '
              'BN_splits {} '
              'long_ind {} *****'.format(lr,
                                         stats[0][0], gamma_tau,
                                         bs[0], bs[1], bs[2],
                                         stats[1][0], stats[2][0], stats[3][0],
                                         bn_splits,
                                         long_ind))


####################################################
# Training pipeline
####################################################
def run(init_lr=init_lr, max_epochs=100, root=npy_root_dir, anno=anno_json_path, batch_size=batch*batch_upscale):

    frames=80 # DOUBLED INSIDE DATASET, AS LONGER CLIPS
    crop_size = {'S':160, 'M':224, 'XL':312}[X3D_VERSION]
    resize_size = {'S':[180.,225.], 'M':[256.,256.], 'XL':[360.,450.]}[X3D_VERSION] # 'M':[256.,320.] FOR LONGER SCHEDULE
    gamma_tau = {'S':6, 'M':5, 'XL':5}[X3D_VERSION] # DOUBLED INSIDE DATASET, AS LONGER CLIPS

    st_steps = 0 # FOR LR WARM-UP
    load_steps = 0 # FOR LOADING AND PRINT SCHEDULE
    steps = 0
    epochs = 0
    num_steps_per_update = 1 # ACCUMULATE GRADIENT IF NEEDED
    cur_iterations = steps * num_steps_per_update
    iterations_per_epoch = TA2_DATASET_SIZE['train']//batch_size
    val_iterations_per_epoch = TA2_DATASET_SIZE['val']//(batch_size//2)
    max_steps = iterations_per_epoch * max_epochs


    train_spatial_transforms = Compose([MultiScaleRandomCropMultigrid([crop_size/i for i in resize_size], crop_size),
                                        RandomHorizontalFlip(),
                                        ToTensor(255),
                                        Normalize(TA2_MEAN, TA2_STD)])

    val_spatial_transforms = Compose([CenterCropScaled(crop_size),
                                        ToTensor(255),
                                        Normalize(TA2_MEAN, TA2_STD)])

    dataset = customized_dataset(split_file=anno,
                                 split='training',
                                 root=root,
                                 num_classes=26,
                                 spatial_transform=train_spatial_transforms,
                                 frames=80,
                                 gamma_tau=gamma_tau,
                                 crops=1,
                                 use_contrastive_loss=False)

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=12,
                                             pin_memory=True)

    val_dataset = customized_dataset(split_file=anno,
                                     split='validation',
                                     root=root,
                                     num_classes=26,
                                     spatial_transform=val_spatial_transforms,
                                     frames=80,
                                     gamma_tau=gamma_tau,
                                     crops=10,
                                     use_contrastive_loss=False)

    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=batch_size//2,
                                                 shuffle=False,
                                                 num_workers=12,
                                                 pin_memory=True)

    dataloaders = {'train': dataloader, 'val': val_dataloader}
    datasets = {'train': dataset, 'val': val_dataset}
    print('train',len(datasets['train']),'val',len(datasets['val']))
    print('Total iterations:', max_steps, 'Total epochs:', max_epochs)
    print('datasets created')


    x3d = resnet_x3d.generate_model(x3d_version=X3D_VERSION, n_classes=400, n_input_channels=3, dropout=0.5, base_bn_splits=1)
    load_ckpt = torch.load('models/x3d_multigrid_kinetics_fb_pretrained.pt')
    x3d.load_state_dict(load_ckpt['model_state_dict'])
    save_model = model_save_dir + '/x3d_ta2_rgb_sgd_'
    # TODO: what is replace_logits
    # x3d.replace_logits(88)
    x3d.replace_logits(101)

    if steps>0:
        load_ckpt = torch.load(model_save_dir + '/x3d_ta2_rgb_sgd_'+str(load_steps).zfill(6)+'.pt')
        x3d.load_state_dict(load_ckpt['model_state_dict'])

    x3d.cuda()
    x3d = nn.DataParallel(x3d)
    print('model loaded')

    lr = init_lr
    print ('INIT LR: %f'%lr)


    optimizer = optim.SGD(x3d.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5)
    lr_sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.1, verbose=True)
    if steps > 0:
        optimizer.load_state_dict(load_ckpt['optimizer_state_dict'])
        lr_sched.load_state_dict(load_ckpt['scheduler_state_dict'])

    criterion = nn.BCEWithLogitsLoss()

    val_apm = APMeter()
    tr_apm = APMeter()
    best_map = 0
    while epochs < max_epochs:
        print ('Step {} Epoch {}'.format(steps, epochs))
        print ('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train']+['val']:
            bar_st = iterations_per_epoch if phase == 'train' else val_iterations_per_epoch
            bar = pkbar.Pbar(name='update: ', target=bar_st)
            if phase == 'train':
                x3d.train(True)
                epochs += 1
                torch.autograd.set_grad_enabled(True)
            else:
                x3d.train(False)  # Set model to evaluate mode
                _ = x3d.module.aggregate_sub_bn_stats() # FOR EVAL AGGREGATE BN STATS
                torch.autograd.set_grad_enabled(False)

            tot_loss = 0.0
            tot_cls_loss = 0.0
            num_iter = 0
            optimizer.zero_grad()

            # Iterate over data.
            print(phase)
            for i,data in enumerate(dataloaders[phase]):
                num_iter += 1
                bar.update(i)
                if phase == 'train':
                    inputs, labels = data
                    print("Checking input and label size")
                    print(inputs.shape)

                else:
                    inputs, labels = data
                    b,n,c,t,h,w = inputs.shape # FOR MULTIPLE TEMPORAL CROPS
                    inputs = inputs.view(b*n,c,t,h,w)

                inputs = inputs.cuda() # B 3 T W H
                labels = labels.cuda() # B C

                if phase == 'train':
                    logits, _, _ = x3d(inputs)
                    logits = logits.squeeze(2) # B C
                    probs = F.sigmoid(logits)
                    # print("Check output size")
                    # print(logits.shape)
                    # print(probs.shape)

                    print("@" * 30)
                    print(logits)
                    print("@" * 30)

                else:
                    with torch.no_grad():
                        logits, _, _ = x3d(inputs)
                    logits = logits.squeeze(2) # B C
                    logits = logits.view(b,n,logits.shape[1]) # FOR MULTIPLE TEMPORAL CROPS
                    probs = F.sigmoid(logits)
                    #probs = torch.mean(probs, 1)
                    #logits = torch.mean(logits, 1)
                    probs = torch.max(probs, dim=1)[0]
                    logits = torch.max(logits, dim=1)[0]


                cls_loss = criterion(logits, labels)
                tot_cls_loss += cls_loss.item()

                if phase == 'train':
                    tr_apm.add(probs.detach().cpu().numpy(), labels.cpu().numpy())
                else:
                    val_apm.add(probs.detach().cpu().numpy(), labels.cpu().numpy())

                loss = cls_loss/num_steps_per_update
                tot_loss += loss.item()

                if phase == 'train':
                    loss.backward()

                if num_iter == num_steps_per_update and phase == 'train':
                    #lr_warmup(lr, steps-st_steps, warmup_steps, optimizer)
                    steps += 1
                    num_iter = 0
                    optimizer.step()
                    optimizer.zero_grad()
                    s_times = iterations_per_epoch//2
                    if (steps-load_steps) % s_times == 0:
                        tr_map = tr_apm.value().mean()
                        tr_apm.reset()
                        print (' Epoch:{} {} steps: {} Cls Loss: {:.4f} Tot Loss: {:.4f} mAP: {:.4f}\n'.format(epochs, phase,
                            steps, tot_cls_loss/(s_times*num_steps_per_update), tot_loss/s_times, tr_map))#, tot_acc/(s_times*num_steps_per_update)))
                        tot_loss = tot_cls_loss = 0.
                    '''if steps % (1000) == 0:
                        ckpt = {'model_state_dict': x3d.module.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'scheduler_state_dict': lr_sched.state_dict()}
                        torch.save(ckpt, save_model+str(steps).zfill(6)+'.pt')'''

            if phase == 'val':
                val_map = val_apm.value().mean()
                lr_sched.step(tot_loss)
                val_apm.reset()
                print (' Epoch:{} {} Loc Cls Loss: {:.4f} Tot Loss: {:.4f} mAP: {:.4f}\n'.format(epochs, phase,
                    tot_cls_loss/num_iter, (tot_loss*num_steps_per_update)/num_iter, val_map))
                tot_loss = tot_cls_loss = 0.
                if val_map > best_map:
                    ckpt = {'model_state_dict': x3d.module.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': lr_sched.state_dict()}
                    best_map = val_map
                    best_epoch = epochs
                    print (' Epoch:{} {} best mAP: {:.4f}\n'.format(best_epoch, phase, best_map))
                    torch.save(ckpt, save_model+'best.pt')



if __name__ == '__main__':
    run()

