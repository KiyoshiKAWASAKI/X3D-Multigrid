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
parser.add_argument('-gpu', default='0', type=str)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu



##################################################################
# Hyper-parameters
##################################################################
work_machine = "kitware"
dataset_used = "hmdb51"
test_known = True
use_feedback = True
update_with_train = True

INIT_LR = 0.001
update_fre = 4

##################################################################
# Hyper-parameters
##################################################################
BS = 2
GPUS = 1
BS_UPSCALE = 2
X3D_VERSION = 'M'
TA2_MEAN = [0, 0, 0]
TA2_STD = [1, 1, 1]

##################################################################
# Data and model path on CRC (No need to change this)
##################################################################
"""
TA2_ANNO: test phase validation set for known samples
TA2_FEEDBACK: test phase feedback set
TA2_UNKNOWN: test phase validation set for unknown samples
"""
if dataset_used == "ucf101":
    nb_classes = 51

    if work_machine == "kitware":
        training_json_path = "/data/jin.huang/ucf101_npy_json/ta2_10_folds/0/ta2_10_folds_partition_0.json"
        TA2_ROOT = '/data/jin.huang/ucf101_npy_json/ta2_10_folds/0'
        trained_model_path = "/data/jin.huang/models/x3d/thresholding/0702_ucf/x3d_ta2_rgb_sgd_best.pt"

        TA2_ANNO = '/data/jin.huang/ucf101_npy_json/ta2_10_folds/0/ta2_partition_0_test_known_test.json'
        TA2_FEEDBACK = '/data/jin.huang/ucf101_npy_json/ta2_10_folds/0/ta2_partition_0_test_known_feedback.json'
        TA2_UNKNOWN = "/data/jin.huang/ucf101_npy_json/ta2_10_folds/0/ta2_partition_0_test_unknown_feedback.json"

    else:
        trainining_json_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/kitware_internship/" \
                               "ucf101_npy_json/ta2_10_folds/0_crc/ta2_10_folds_partition_0.json"
        TA2_ROOT = '/afs/crc.nd.edu/user/j/jhuang24/scratch_51/kitware_internship/ucf101_npy_json/ta2_10_folds/0_crc'
        trained_model_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/kitware_internship" \
                             "/models/x3d/thresholding/0729_ucf/x3d_ta2_rgb_sgd_best.pt"

        TA2_ANNO = '/afs/crc.nd.edu/user/j/jhuang24/scratch_51/kitware_internship' \
                   '/ucf101_npy_json/ta2_10_folds/0_crc/ta2_partition_0_test_known_test.json'
        TA2_FEEDBACK = '/afs/crc.nd.edu/user/j/jhuang24/scratch_51/kitware_internship/' \
                       'ucf101_npy_json/ta2_10_folds/0_crc/ta2_partition_0_test_known_feedback.json'
        TA2_UNKNOWN = '/afs/crc.nd.edu/user/j/jhuang24/scratch_51/kitware_internship' \
                       '/ucf101_npy_json/ta2_10_folds/0_crc/ta2_partition_0_test_unknown_feedback.json'


else:
    nb_classes = 26
    threshold = [0.05, 0.06, 0.07, 0.08, 0.09, 0.10]

    if work_machine == "kitware":
        training_json_path = "/data/jin.huang/hmdb51/npy_json/0/ta2_10_folds_partition_0.json"
        TA2_ROOT = "/data/jin.huang/hmdb51/npy_json/0"
        trained_model_path = "/data/jin.huang/models/x3d/thresholding/0702_hmdb/x3d_ta2_rgb_sgd_best.pt"

        TA2_ANNO = "/data/jin.huang/hmdb51/npy_json/0/ta2_partition_0_test_known_test.json"
        TA2_FEEDBACK = "/data/jin.huang/hmdb51/npy_json/0/ta2_partition_0_test_known_feedback.json"
        TA2_UNKNOWN = "/data/jin.huang/hmdb51/npy_json/0/ta2_partition_0_test_unknown_test.json"

    else:
        trainining_json_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/kitware_internship/hmdb51/npy_json/" \
                               "0_crc/ta2_10_folds_partition_0.json"
        TA2_ROOT = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/kitware_internship/hmdb51/npy_json/0_crc"
        trained_model_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/kitware_internship/models/" \
                             "x3d/thresholding/0802_hmdb/x3d_ta2_rgb_sgd_best.pt"

        TA2_ANNO = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/kitware_internship" \
                   "/hmdb51/npy_json/0_crc/ta2_partition_0_test_known_test.json"
        TA2_FEEDBACK = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/kitware_internship/" \
                       "hmdb51/npy_json/0_crc/ta2_partition_0_test_known_feedback.json"
        TA2_UNKNOWN = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/kitware_internship/" \
                       "hmdb51/npy_json/0_crc/ta2_partition_0_test_unknown_test.json"


##################################################################
# Data and model path on CRC (No need to change this)
##################################################################
def run(init_lr,
        root,
        anno,
        feedback_file_known,
        test_unknown_file,
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
    train_dataset = UCF101(split_file=training_json_path,
                           split='training',
                           root=root,
                           num_classes=nb_classes,
                           spatial_transform=train_spatial_transforms,
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
    feedback_dataset = UCF101(split_file=feedback_file_known,
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

    # Test validation data loader - known
    val_dataset_known = UCF101(split_file=anno,
                         split='test_set',
                         root=root,
                         num_classes=nb_classes,
                         spatial_transform=val_spatial_transforms,
                         frames=80,
                         gamma_tau=gamma_tau,
                         crops=10,
                         test_phase=True,
                         is_feedback=True)
    val_dataloader_known = torch.utils.data.DataLoader(val_dataset_known,
                                                 batch_size=batch_size // 2,
                                                 shuffle=False,
                                                 num_workers=4,
                                                 pin_memory=True)

    # Test validation data loader - unknown
    val_dataset_unknown = UCF101(split_file=test_unknown_file,
                               split='test_set',
                               root=root,
                               num_classes=nb_classes,
                               spatial_transform=val_spatial_transforms,
                               frames=80,
                               gamma_tau=gamma_tau,
                               crops=10,
                               test_phase=True,
                               is_feedback=True)
    val_dataloader_unknown = torch.utils.data.DataLoader(val_dataset_unknown,
                                                       batch_size=batch_size // 2,
                                                       shuffle=False,
                                                       num_workers=4,
                                                       pin_memory=True)


    #################################################
    # Update model
    #################################################
    # Use trained model
    x3d = resnet_x3d.generate_model(x3d_version=X3D_VERSION,
                                    n_classes=nb_classes,
                                    n_input_channels=3,
                                    dropout=0.5,
                                    base_bn_splits=1)
    load_ckpt = torch.load(trained_model_path)
    x3d.load_state_dict(load_ckpt['model_state_dict'])
    x3d.cuda()

    # Freeze some layers
    for name, param in x3d.named_parameters():
        if param.requires_grad and \
                name != "fc2.weight" and name != "fc2.bias" \
                and name != "fc1.weight":
            param.requires_grad = False

    # Training setups
    x3d = nn.DataParallel(x3d)
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, x3d.parameters()), lr=init_lr, momentum=0.9, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    val_apm = APMeter()

    sm = torch.nn.Softmax(dim=1)

    # Update network/fine-tuning
    train_dataloader_iterator = iter(train_dataloader)

    for i, data in enumerate(feedback_dataloader):
        if dataset_used == "hmdb51":
            count_known_list = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
            count_unknown_list = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
        else:
            pass

        data_feedback, labels_feedback = data

        try:
            data_train, labels_train = next(train_dataloader_iterator)
        except StopIteration:
            train_dataloader_iterator = iter(train_dataloader)
            data_train, labels_train = next(train_dataloader_iterator)

        # Just update the network every 4 steps
        # if (i == 3) or (i % 4 == 3):
        x3d.train(True)
        torch.autograd.set_grad_enabled(True)
        optimizer.zero_grad()

        # Get data from training set and feedback set
        b_fb, n_fb, c_fb, t_fb, h_fb, w_fb = data_feedback.shape
        data_feedback = data_feedback.view(b_fb * n_fb, c_fb, t_fb, h_fb, w_fb)

        b_train, n_train, c_train, t_train, h_train, w_train = data_train.shape
        data_train = data_train.view(b_train * n_train, c_train, t_train, h_train, w_train)

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

        if update_with_train:
            logits = logits.view(b_fb, n_fb, logits.shape[1])
        else:
            logits = logits.view(b_fb*2, n_fb, logits.shape[1])

        logits = torch.max(logits, dim=1)[0]

        loss = criterion(logits, labels)
        loss.backward()

        #################################################
        # Testing known samples
        #################################################
        # Switch to validation Mode
        x3d.train(False)
        _ = x3d.module.aggregate_sub_bn_stats()
        torch.autograd.set_grad_enabled(False)

        # TODO: Test known samples
        progress_bar_known = pkbar.Pbar(name="Testing known:",
                                  target=len(val_dataloader_known))

        for k, data in enumerate(val_dataloader_known):
            progress_bar_known.update(k)
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

            # TODO: Get the probabilities
            probs_sm = sm(probs)

            for one_prob in probs_sm:
                max_prob = torch.max(one_prob)
                # print(max_prob)

                for t in range(len(threshold)):
                    one_thresh = threshold[t]

                    if max_prob > one_thresh:
                        count_known_list[t][0] += 1
                    else:
                        count_known_list[t][1] += 1

            val_apm.add(probs.detach().cpu().numpy(), labels.cpu().numpy())

        val_map = val_apm.value().mean()
        print("All counts:", count_known_list)
        print('Epoch (testing):val mAP: {:.4f}'.format(val_map))

        #################################################
        # Testing unknown samples
        ################################################
        progress_bar_unknown = pkbar.Pbar(name="Testing unknown:",
                                        target=len(val_dataloader_unknown))

        for k, data in enumerate(val_dataloader_unknown):
            progress_bar_unknown.update(k)
            inputs, labels = data

            b, n, c, t, h, w = inputs.shape  # FOR MULTIPLE TEMPORAL CROPS
            inputs = inputs.view(b * n, c, t, h, w)

            inputs = inputs.cuda()  # B 3 T W H

            with torch.no_grad():
                logits, feat, base = x3d(inputs)

            logits = logits.squeeze(2)  # B C
            logits = logits.view(b, n, logits.shape[1])

            probs = F.sigmoid(logits)
            probs = torch.max(probs, dim=1)[0]
            logits = torch.max(logits, dim=1)[0]

            # TODO: Get the probabilities
            probs_sm = sm(probs)

            for one_prob in probs_sm:
                max_prob = torch.max(one_prob)
                # print(max_prob)

                for t in range(len(threshold)):
                    one_thresh = threshold[t]

                    if max_prob > one_thresh:
                        count_unknown_list[t][0] += 1
                    else:
                        count_unknown_list[t][1] += 1

        print("Unknown counts:", count_unknown_list)


if __name__ == '__main__':
    run(root=TA2_ROOT,
        anno=TA2_ANNO,
        feedback_file_known=TA2_FEEDBACK,
        test_unknown_file=TA2_UNKNOWN,
        batch_size=BS*BS_UPSCALE,
        init_lr=INIT_LR)
