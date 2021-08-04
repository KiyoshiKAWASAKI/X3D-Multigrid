import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import logging

import torchvision
from torchvision import datasets, transforms
from torchsummary import summary

import numpy as np
from barbar import Bar
import pkbar
from utils.apmeter import APMeter

import x3d as resnet_x3d

from data.ucf101 import customized_dataset, UCF101

from transforms.spatial_transforms_old import Compose, Normalize, \
    RandomHorizontalFlip, MultiScaleRandomCrop, \
    MultiScaleRandomCropMultigrid, ToTensor, \
    CenterCrop, CenterCropScaled
from transforms.temporal_transforms import TemporalRandomCrop
from transforms.target_transforms import ClassLabel
import pdb

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('-gpu', default='1', type=str)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu



##################################################################
# Hyper-parameters
##################################################################
BS = 2
BS_UPSCALE = 2
INIT_LR = 0.00005 * BS_UPSCALE
GPUS = 1

dataset_used = "ucf101"
test_known = True
use_feedback = True
update_with_train = True

threshold = [0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
update_fre = 4

##################################################################
# Data and model path on CRC (No need to change this)
##################################################################

X3D_VERSION = 'M'
TA2_MEAN = [0, 0, 0]
TA2_STD = [1, 1, 1]

if dataset_used == "ucf101":
    nb_classes = 51

    trainining_json_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/kitware_internship/" \
                           "ucf101_npy_json/ta2_10_folds/0_crc/ta2_10_folds_partition_0.json"
    TA2_ROOT = '/afs/crc.nd.edu/user/j/jhuang24/scratch_51/kitware_internship/ucf101_npy_json/ta2_10_folds/0_crc'
    trained_model_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/kitware_internship" \
                         "/models/x3d/thresholding/0729_ucf/x3d_ta2_rgb_sgd_best.pt"

    if test_known == True:
        if use_feedback == False:
            TA2_ANNO = '/afs/crc.nd.edu/user/j/jhuang24/scratch_51/kitware_internship' \
                       '/ucf101_npy_json/ta2_10_folds/0_crc/ta2_partition_0_test_known_test.json'
            TA2_DATASET_SIZE = {'train': None, 'val': 1026}
        else:
            TA2_ANNO = '/afs/crc.nd.edu/user/j/jhuang24/scratch_51/kitware_internship' \
                       '/ucf101_npy_json/ta2_10_folds/0_crc/ta2_partition_0_test_known_test.json'
            TA2_FEEDBACK = '/afs/crc.nd.edu/user/j/jhuang24/scratch_51/kitware_internship/' \
                           'ucf101_npy_json/ta2_10_folds/0_crc/ta2_partition_0_test_known_feedback.json'
            TA2_DATASET_SIZE = {'train': None, 'val': 1026}

    else:
        if use_feedback == False:
            TA2_ANNO = '/afs/crc.nd.edu/user/j/jhuang24/scratch_51/kitware_internship' \
                       '/ucf101_npy_json/ta2_10_folds/0_crc/ta2_partition_0_test_unknown_test.json'
            TA2_DATASET_SIZE = {'train': None, 'val': 1026}
        else:
            TA2_ANNO = '/afs/crc.nd.edu/user/j/jhuang24/scratch_51/kitware_internship' \
                       '/ucf101_npy_json/ta2_10_folds/0_crc/ta2_partition_0_test_unknown_test.json'
            TA2_FEEDBACK = '/afs/crc.nd.edu/user/j/jhuang24/scratch_51/kitware_internship' \
                           '/ucf101_npy_json/ta2_10_folds/0_crc/ta2_partition_0_test_unknown_feedback.json'
            TA2_DATASET_SIZE = {'train': None, 'val': 1026}


else:
    nb_classes = 26

    trainining_json_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/kitware_internship/hmdb51/npy_json/" \
                           "0_crc/ta2_10_folds_partition_0.json"
    TA2_ROOT = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/kitware_internship/hmdb51/npy_json/0_crc"
    trained_model_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/kitware_internship/models/" \
                         "x3d/thresholding/0802_hmdb/x3d_ta2_rgb_sgd_best.pt"

    if test_known == True:
        if use_feedback == False:
            TA2_ANNO = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/kitware_internship/" \
                       "hmdb51/npy_json/0_crc/ta2_partition_0_test_known_test.json"
            TA2_DATASET_SIZE = {'train': None, 'val': 464}
        else:
            TA2_ANNO = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/kitware_internship" \
                       "/hmdb51/npy_json/0_crc/ta2_partition_0_test_known_test.json"
            TA2_FEEDBACK = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/kitware_internship/" \
                           "hmdb51/npy_json/0_crc/ta2_partition_0_test_known_feedback.json"
            TA2_DATASET_SIZE = {'train': None, 'val': 464}
    else:
        if use_feedback == False:
            TA2_ANNO = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/kitware_internship/" \
                       "hmdb51/npy_json/0_crc/ta2_partition_0_test_unknown_test.json"
            TA2_DATASET_SIZE = {'train': None, 'val': 464}
        else:
            TA2_ANNO = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/kitware_internship/" \
                       "hmdb51/npy_json/0_crc/ta2_partition_0_test_unknown_test.json"
            TA2_FEEDBACK = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/kitware_internship/" \
                           "hmdb51/npy_json/0_crc/ta2_partition_0_test_unknown_test.json"
            TA2_DATASET_SIZE = {'train': None, 'val': 464}


##################################################################
# Data and model path on CRC (No need to change this)
##################################################################
def run(init_lr,
        root,
        anno,
        feedback_file,
        batch_size=BS*BS_UPSCALE):

    frames=80 # DOUBLED INSIDE DATASET, AS LONGER CLIPS
    crop_size = {'S':160, 'M':224, 'XL':312}[X3D_VERSION]
    resize_size = {'S':[180.,225.], 'M':[256.,256.], 'XL':[360.,450.]}[X3D_VERSION] # 'M':[256.,320.] FOR LONGER SCHEDULE
    gamma_tau = {'S':6, 'M':5, 'XL':5}[X3D_VERSION] # DOUBLED INSIDE DATASET, AS LONGER CLIPS

    ##################################################################
    # Data Loaders
    ##################################################################
    train_spatial_transforms = Compose([MultiScaleRandomCropMultigrid([crop_size/i for i in resize_size], crop_size),
                                        RandomHorizontalFlip(),
                                        ToTensor(255),
                                        Normalize(TA2_MEAN, TA2_STD)])

    val_spatial_transforms = Compose([CenterCropScaled(crop_size),
                                        ToTensor(255),
                                        Normalize(TA2_MEAN, TA2_STD)])
    # Training data loader
    train_dataset = UCF101(split_file=trainining_json_path,
                           split='training',
                           root=root,
                           num_classes=nb_classes,
                           spatial_transform=val_spatial_transforms,
                           frames=80,
                           gamma_tau=gamma_tau,
                           crops=10,
                           test_phase=True,
                           is_feedback=True)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size // 2,
                                                   shuffle=False,
                                                   num_workers=0,
                                                   pin_memory=True)

    # Feedback data loader
    feedback_dataset = UCF101(split_file=feedback_file,
                              split='feedback_set',
                              root=root,
                              num_classes=nb_classes,
                              spatial_transform=val_spatial_transforms,
                              frames=80,
                              gamma_tau=gamma_tau,
                              crops=10,
                              test_phase=True,
                              is_feedback=True)
    feedback_dataloader = torch.utils.data.DataLoader(feedback_dataset,
                                                      batch_size=batch_size // 2,
                                                      shuffle=False,
                                                      num_workers=4,
                                                      pin_memory=True)

    # Test validation data loader
    val_dataset = UCF101(split_file=anno,
                         split='test_set',
                         root=root,
                         num_classes=nb_classes,
                         spatial_transform=val_spatial_transforms,
                         frames=80,
                         gamma_tau=gamma_tau,
                         crops=10,
                         test_phase=True,
                         is_feedback=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=batch_size // 2,
                                                 shuffle=False,
                                                 num_workers=4,
                                                 pin_memory=True)

    ##################################################################
    # Update model
    ##################################################################
    # Use trained model
    x3d = resnet_x3d.generate_model(x3d_version=X3D_VERSION,
                                    n_classes=nb_classes,
                                    n_input_channels=3,
                                    dropout=0.5,
                                    base_bn_splits=1)
    load_ckpt = torch.load(trained_model_path)
    x3d.load_state_dict(load_ckpt['model_state_dict'])

    x3d.cuda()
    x3d = nn.DataParallel(x3d)

    lr = init_lr
    print ('INIT LR: %f'%lr)


    optimizer = optim.SGD(x3d.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5)
    lr_sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.1, verbose=True)
    criterion = nn.BCEWithLogitsLoss()
    val_apm = APMeter()


    # Update network and do test
    train_dataloader_iterator = iter(train_dataloader)

    for i, data in enumerate(feedback_dataloader):
        data_feedback, labels_feedback = data

        try:
            data_train, labels_train = next(train_dataloader_iterator)
        except StopIteration:
            train_dataloader_iterator = iter(train_dataloader)
            data_train, labels_train = next(train_dataloader_iterator)

        # Just update the network every 4 steps
        if (i == 3) or (i % 4 == 3):
            # TODO: set the model on training
            x3d.train(True)
            torch.autograd.set_grad_enabled(True)
            optimizer.zero_grad()

            # Get data from training set and feedback set
            b_fb, n_fb, c_fb, t_fb, h_fb, w_fb = data_feedback.shape
            data_feedback = data_feedback.view(b_fb * n_fb, c_fb, t_fb, h_fb, w_fb)

            b_train, n_train, c_train, t_train, h_train, w_train = data_train.shape
            data_train = data_train.view(b_train * n_train, c_train, t_train, h_train, w_train)

            # TODO: Get the data for training
            if update_with_train:
                inputs = data_train
                labels = labels_train

            else:
                inputs = torch.cat((data_feedback, data_train), dim=0)
                labels = torch.cat((labels_feedback, labels_train), dim=0)

            inputs = inputs.cuda()
            labels = labels.cuda()

            # TODO: Updating network
            print("Updating network.")
            logits, _, _ = x3d(inputs)
            logits = logits.squeeze(2)
            # probs = F.sigmoid(logits)

            loss = criterion(logits, labels)
            loss.backward()

            # TODO: Then, do validation
            x3d.train(False)
            _ = x3d.module.aggregate_sub_bn_stats()
            torch.autograd.set_grad_enabled(False)

            progress_bar = pkbar.Pbar(name="Testing the whole validation set:",
                                      target=len(val_dataloader))

            for k, data in enumerate(val_dataloader):
                progress_bar.update(k)
                inputs, labels = data

                b, n, c, t, h, w = inputs.shape  # FOR MULTIPLE TEMPORAL CROPS
                inputs = inputs.view(b * n, c, t, h, w)

                inputs = inputs.cuda()  # B 3 T W H
                labels = labels.cuda()  # B C

                with torch.no_grad():
                    logits, feat, base = x3d(inputs)

                logits = logits.squeeze(2)  # B C
                logits = logits.view(b, n, logits.shape[1])

                probs = F.sigmoid(logits)
                probs = torch.max(probs, dim=1)[0]
                logits = torch.max(logits, dim=1)[0]

                val_apm.add(probs.detach().cpu().numpy(), labels.cpu().numpy())

            val_map = val_apm.value().mean()
            print('Epoch (testing):{} val mAP: {:.4f}'.format(epochs, val_map))

        else:
            continue



if __name__ == '__main__':
    run(root=TA2_ROOT,
      anno=TA2_ANNO,
      feedback_file=TA2_FEEDBACK,
      batch_size=BS*BS_UPSCALE,
      init_lr=INIT_LR)
