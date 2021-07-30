# Training X3D with contrastive loss
# Modified based on Dawei's code and original X3D by Jin Huang

import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler

from torchsummary import summary

from barbar import Bar
import pkbar
from utils.apmeter import APMeter

import x3d as resnet_x3d
from data.ucf101 import customized_dataset
from transforms.spatial_transforms import Compose, Normalize, RandomHorizontalFlip, MultiScaleRandomCropMultigrid, ToTensor, CenterCrop, CenterCropScaled
from transforms.temporal_transforms import TemporalRandomCrop
from transforms.target_transforms import ClassLabel

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('-gpu', default='0', type=str)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

# TODO: Rewrite and debug the whole code so that we can train it on 3 datasets
# TODO: Integrate CC into X3D training pipeline
# TODO: Need to build a whole new testing pipeline

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



###############################################################
# Define some utility functions
###############################################################
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
        print(' ***** LR {} Frames {}/{} '
              'BS ({},{}) W/H ({},{}) '
              'BN_splits {} '
              'long_ind {} *****'.format(lr, stats[0][0], gamma_tau,
                                         bs[0], bs[1], stats[2][0], stats[3][0],
                                         bn_splits,
                                         long_ind))

    else:
        bs = [bs*j for j in [4,2,1]]
        print(' ***** LR {} Frames {}/{} '
              'BS ({},{},{}) '
              'W/H ({},{},{}) '
              'BN_splits {} '
              'long_ind {} *****'.format(lr, stats[0][0], gamma_tau,
                                         bs[0], bs[1], bs[2],
                                         stats[1][0], stats[2][0], stats[3][0],
                                         bn_splits,
                                         long_ind))



###############################################################
# Define data transformation and get dataloader
###############################################################

# Define spatial transformation
train_spatial_transforms = Compose([MultiScaleRandomCropMultigrid([crop_size/i for i in resize_size], crop_size),
                                        RandomHorizontalFlip(),
                                        ToTensor(255),
                                        Normalize(data_mean, data_std)])

val_spatial_transforms = Compose([CenterCropScaled(crop_size),
                                        ToTensor(255),
                                        Normalize(data_mean, data_std)])

# Get dataloaders
train_dataset = customized_dataset(split_file=anno_json_path,
                                     split='training',
                                     root=npy_root_dir,
                                     num_classes=nb_classes,
                                     spatial_transform=train_spatial_transforms,
                                     frames=frames,
                                     gamma_tau=gamma_tau,
                                     crops=1)


val_dataset = customized_dataset(split_file=anno_json_path,
                                 split='validation',
                                 root=npy_root_dir,
                                 num_classes=nb_classes,
                                 spatial_transform=val_spatial_transforms,
                                 frames=frames,
                                 gamma_tau=gamma_tau,
                                 crops=10)

train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                 batch_size=batch * batch_upscale,
                                                 shuffle=True,
                                                 num_workers=nb_workers,
                                                 pin_memory=True)

val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch,
                                             shuffle=False,
                                             num_workers=nb_workers,
                                             pin_memory=True)



###############################################################
# Main function for training
###############################################################
def run(init_lr,
        max_epochs,
        batch_size):

    print("[INFO] There are %d training batches and %d validation batches"
          % (len(train_dataloader), len(val_dataloader)))
    print("[INFO] Initial learning rate: %f" % init_lr)

    # Create the model
    x3d = resnet_x3d.generate_model(x3d_version=X3D_VERSION,
                                    n_classes=400,
                                    n_input_channels=3,
                                    dropout=0.5,
                                    base_bn_splits=1)

    optimizer = optim.SGD(x3d.parameters(),
                          lr=init_lr,
                          momentum=0.9,
                          weight_decay=1e-5)

    lr_sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                    mode='min',
                                                    patience=2,
                                                    factor=0.1,
                                                    verbose=True)

    val_apm = APMeter()
    tr_apm = APMeter()
    criterion = nn.BCEWithLogitsLoss()

    best_map = 0.0

    # If using pretrain, load model and change the last layer
    if use_pretrain:
        print("Using pretrain model.")
        load_ckpt = torch.load(pretrain_model_path)
        x3d.load_state_dict(load_ckpt['model_state_dict'])
        x3d.replace_logits(nb_classes)

    if cont_training:
        load_ckpt = torch.load(previous_model_path)
        x3d.load_state_dict(load_ckpt['model_state_dict'])
        optimizer.load_state_dict(load_ckpt['optimizer_state_dict'])
        lr_sched.load_state_dict(load_ckpt['scheduler_state_dict'])

    x3d.cuda()
    x3d = nn.DataParallel(x3d)

    # Run training and validation in turns
    for e in range(max_epochs):
        # Training
        train_bar = pkbar.Pbar(name="Training:",
                               target=len(train_dataloader))

        x3d.train(True)
        torch.autograd.set_grad_enabled(True)
        optimizer.zero_grad()

        tot_loss = 0.0
        tot_cls_loss = 0.0

        for i, data in enumerate(train_dataloader):
            train_bar.update(i)

            inputs, labels = data
            inputs = inputs.cuda()  # B 3 T W H
            labels = labels.cuda()  # B C

            # print(inputs.shape)
            # print(labels.shape)

            logits, _, _ = x3d(inputs)
            logits = logits.squeeze(2)  # B C
            probs = F.sigmoid(logits)

            cls_loss = criterion(logits, labels)
            tot_cls_loss += cls_loss.item()

            tr_apm.add(probs.detach().cpu().numpy(), labels.cpu().numpy())

            loss = cls_loss
            tot_loss += loss.item()

            # cls_loss.backward()
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            s_times = len(train_dataloader) // 2

            # Print mAP at the end of each epoch
            if i == len(train_dataloader) - 1:
                tr_map = tr_apm.value().mean()
                tr_apm.reset()

                print(tr_map)

                # print('Epoch (training):{}/{} '
                #       'Cls Loss: {:.4f} '
                #       'Tot Loss: {:.4f} '
                #       'mAP: {:.4f}\n'.format(e, nb_epoch,
                #                               cls_loss/s_times,
                #                               tot_loss,
                #                               tr_map))


        # # validation
        # valid_bar = pkbar.Pbar(name="Validation:",
        #                        target=len(val_dataloader))
        #
        # x3d.train(False)  # Set model to evaluate mode
        # _ = x3d.module.aggregate_sub_bn_stats() # FOR EVAL AGGREGATE BN STATS
        # torch.autograd.set_grad_enabled(False)
        #
        # tot_loss = 0.0
        # tot_cls_loss = 0.0
        # # optimizer.zero_grad()
        #
        # for i, data in enumerate(val_dataloader):
        #     valid_bar.update(i)
        #     inputs, labels = data
        #     b, n, c, t, h, w = inputs.shape  # FOR MULTIPLE TEMPORAL CROPS
        #
        #     inputs = inputs.view(b * n, c, t, h, w)
        #     inputs = inputs.cuda() # B 3 T W H
        #     labels = labels.cuda() # B C
        #
        #     with torch.no_grad():
        #         logits, _, _ = x3d(inputs)
        #
        #     logits = logits.squeeze(2) # B C
        #     logits = logits.view(b,n,logits.shape[1]) # FOR MULTIPLE TEMPORAL CROPS
        #     probs = F.sigmoid(logits)
        #
        #     probs = torch.max(probs, dim=1)[0]
        #     logits = torch.max(logits, dim=1)[0]
        #
        #     cls_loss = criterion(logits, labels)
        #     tot_cls_loss += cls_loss.item()
        #
        #     loss = cls_loss
        #     tot_loss += loss.item()
        #
        #     val_apm.add(probs.detach().cpu().numpy(), labels.cpu().numpy())
        #     val_map = val_apm.value().mean()
        #
        #     lr_sched.step(tot_loss)
        #     val_apm.reset()
        #
        #     if i == len(val_dataloader) - 1:
        #         print ('Epoch (validation):{}/{} '
        #                'Loc Cls Loss: {:.4f} '
        #                'Tot Loss: {:.4f} '
        #                'mAP: {:.4f}\n'.format(e, nb_epoch,
        #                                       tot_cls_loss,
        #                                       tot_loss/i,
        #                                       val_map))
        #
        #         tot_loss = tot_cls_loss = 0.
        #
        #     # Save model if the validation mAP improves
        #     if val_map > best_map:
        #         ckpt = {'model_state_dict': x3d.module.state_dict(),
        #                 'optimizer_state_dict': optimizer.state_dict(),
        #                 'scheduler_state_dict': lr_sched.state_dict()}
        #
        #         best_map = val_map
        #         torch.save(ckpt, os.path.join(model_save_dir, 'best_model.pt'))
        #
        #         print (' Epoch:{}. '
        #                'Current best mAP: {:.4f}\n'.format(e, best_map))


if __name__ == '__main__':
    run(init_lr=init_lr,
        max_epochs=nb_epoch,
        batch_size=batch*batch_upscale)
