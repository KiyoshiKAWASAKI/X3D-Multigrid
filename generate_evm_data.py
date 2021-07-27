import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

import pkbar
# from utils.apmeter import APMeter

import x3d as resnet_x3d

from data.ucf101 import UCF101
import torch.optim as optim

from transforms.spatial_transforms import Compose, Normalize, ToTensor, \
    CenterCropScaled
import sys
import numpy as np

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('-gpu', default='0', type=str)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu


TA2_MEAN = [0, 0, 0]
TA2_STD = [1, 1, 1]

BS = 4
BS_UPSCALE = 2
INIT_LR = 0.02 * BS_UPSCALE
GPUS = 1
X3D_VERSION = 'M'

##################################################################
# UCF 101
ucf101_base_dir = "/data/jin.huang/ucf101_npy_json/ta2_10_folds/0/"
ucf_model_path = "/data/jin.huang/models/x3d/thresholding/0702_ucf/x3d_ta2_rgb_sgd_best.pt"

# Training and validation set for training phase
ucf_101_train_known_json_path = ucf101_base_dir + "ta2_10_folds_partition_0.json"

# Test set (feedbackset + validation set)
ucf_101_test_known_valid_json = ucf101_base_dir + "ta2_partition_0_test_known_test.json"
ucf_101_test_known_feedback_json = ucf101_base_dir + "ta2_partition_0_test_known_feedback.json"
ucf_101_test_unknown_valid_json = ucf101_base_dir + "ta2_partition_0_test_unknown_test.json"
ucf_101_test_unknown_feedback_json = ucf101_base_dir + "ta2_partition_0_test_unknown_feedback.json"

# HMDB 51
hmdb51_base_dir = "/data/jin.huang/hmdb51/npy_json/0/"
hmdb_model_path = "/data/jin.huang/models/x3d/thresholding/0702_hmdb/x3d_ta2_rgb_sgd_best.pt"

# Training and validation set for training phase
hmdb51_train_known_json_path = hmdb51_base_dir + "ta2_partition_0.json"

# Test set (feedbackset + validation set)
hmdb51_test_known_valid_json_path = hmdb51_base_dir + "ta2_partition_0_test_known_test.json"
hmdb51_test_known_feedback_json_path = hmdb51_base_dir + "ta2_partition_0_test_known_feedback.json"
hmdb51_test_unknown_valid_json_path = hmdb51_base_dir + "ta2_partition_0_test_unknown_test.json"
hmdb51_test_unknown_feedback_json_path = hmdb51_base_dir + "ta2_partition_0_test_unknown_feedback.json"

##################################################################
def run(root,
        anno,
        batch_size,
        split,
        trained_model_path,
        nb_classes,
        feature_save_path,
        label_save_path):

    frames=80
    crop_size = {'S':160,
                 'M':224,
                 'XL':312}[X3D_VERSION]
    resize_size = {'S':[180.,225.],
                   'M':[256.,256.],
                   'XL':[360.,450.]}[X3D_VERSION]
    gamma_tau = {'S':6,
                 'M':5,
                 'XL':5}[X3D_VERSION]


    val_spatial_transforms = Compose([CenterCropScaled(crop_size),
                                        ToTensor(255),
                                        Normalize(TA2_MEAN, TA2_STD)])


    print("split in run ", split)

    val_dataset = UCF101(split_file=anno,
                         split=split,
                         root=root,
                         num_classes=nb_classes,
                         spatial_transform=val_spatial_transforms,
                         frames=80,
                         gamma_tau=gamma_tau,
                         crops=10,
                         test_phase=True,
                         is_feedback=False)
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=batch_size // 2,
                                                 shuffle=False,
                                                 num_workers=0,
                                                 pin_memory=True)

    x3d = resnet_x3d.generate_model(x3d_version=X3D_VERSION,
                                    n_classes=nb_classes,
                                    n_input_channels=3,
                                    dropout=0.5,
                                    base_bn_splits=1)

    val_iterations_per_epoch = len(val_dataloader)

    load_ckpt = torch.load(trained_model_path)
    x3d.load_state_dict(load_ckpt['model_state_dict'])

    x3d.cuda()
    x3d = nn.DataParallel(x3d)
    print('model loaded')

    print ('-' * 10)
    bar_st = val_iterations_per_epoch
    bar = pkbar.Pbar(name='Running data thru trained model for EVM ', target=bar_st)
    x3d.train(False)
    _ = x3d.module.aggregate_sub_bn_stats()

    # TODO: Create empty lists for saving all the features and labels
    all_features_list = []
    all_labels_list = []

    # Iterate over data.
    for i, data in enumerate(val_dataloader):
    # for i in range(5):
        bar.update(i)

        # inputs, labels = next(iter(val_dataloader))
        inputs, labels = data

        # TODO: Convert the label to integer and save them into the array
        for i in range(labels.shape[0]):
            one_hot_label = labels[i, :]
            one_label = torch.argmax(one_hot_label)
            print(one_label)
            all_labels_list.append(one_label.item())

        b,n,c,t,h,w = inputs.shape # FOR MULTIPLE TEMPORAL CROPS
        inputs = inputs.view(b*n,c,t,h,w)

        inputs = inputs.cuda() # B 3 T W H
        # labels = labels.cuda() # B C

        with torch.no_grad():
            logits, feat, base = x3d(inputs)

        logits = logits.squeeze(2) # B C
        logits = logits.view(b,n,logits.shape[1])
        # This logits after max is what we want
        logits = torch.max(logits, dim=1)[0]
        # Logits shape: [batch_size, nb_classes]

        # TODO: Append each result into list
        for j in range(logits.shape[0]):
            one_logit = logits[j, :]
            all_features_list.append(one_logit.tolist())

        print("one feature", len(all_features_list[0]))

    #TODO: Convert labels into numpy
    all_label_np = np.asarray(all_labels_list)
    all_label = np.reshape(all_label_np, (len(all_labels_list), 1))

    # TODO: Convert features into numpy
    all_features_np = np.array(all_features_list)

    print("\n")
    print("all feature shape: ", all_features_np.shape)
    print("all label shape: ", all_label.shape)

    np.save(feature_save_path, all_features_np)
    np.save(label_save_path, all_label_np)
    print("NPY files saved")

##################################################################
# HMDB51: training
run(root=hmdb51_base_dir,
    anno=hmdb51_train_known_json_path,
    batch_size=BS*BS_UPSCALE,
    split='training',
    trained_model_path=hmdb_model_path,
    nb_classes=26,
    feature_save_path="/data/jin.huang/hmdb51/npy_json/0/evm_npy/hmdb_train_known_feature.npy",
    label_save_path="/data/jin.huang/hmdb51/npy_json/0/evm_npy/hmdb_train_known_label.npy")

# HMDB51: test known feedback
run(root=hmdb51_base_dir,
    anno=hmdb51_test_known_feedback_json_path,
    batch_size=BS*BS_UPSCALE,
    split='feedback_set',
    trained_model_path=hmdb_model_path,
    nb_classes=26,
    feature_save_path="/data/jin.huang/hmdb51/npy_json/0/evm_npy/hmdb_test_known_feedback_feature.npy",
    label_save_path="/data/jin.huang/hmdb51/npy_json/0/evm_npy/hmdb_test_known_feedback_label.npy")

# HMDB51: test known valid
run(root=hmdb51_base_dir,
    anno=hmdb51_test_known_valid_json_path,
    batch_size=BS*BS_UPSCALE,
    split='test_set',
    trained_model_path=hmdb_model_path,
    nb_classes=26,
    feature_save_path="/data/jin.huang/hmdb51/npy_json/0/evm_npy/hmdb_test_known_valid_feature.npy",
    label_save_path="/data/jin.huang/hmdb51/npy_json/0/evm_npy/hmdb_test_known_valid_label.npy")

# HMDB51: training
run(root=hmdb51_base_dir,
    anno=hmdb51_test_unknown_feedback_json_path,
    batch_size=BS*BS_UPSCALE,
    split='feedback_set',
    trained_model_path=hmdb_model_path,
    nb_classes=26,
    feature_save_path="/data/jin.huang/hmdb51/npy_json/0/evm_npy/hmdb_test_unknown_feedback_feature.npy",
    label_save_path="/data/jin.huang/hmdb51/npy_json/0/evm_npy/hmdb_test_unknown_feedback_label.npy")

# HMDB51: training
run(root=hmdb51_base_dir,
    anno=hmdb51_test_unknown_valid_json_path,
    batch_size=BS*BS_UPSCALE,
    split='test_set',
    trained_model_path=hmdb_model_path,
    nb_classes=26,
    feature_save_path="/data/jin.huang/hmdb51/npy_json/0/evm_npy/hmdb_test_unknown_valid_feature.npy",
    label_save_path="/data/jin.huang/hmdb51/npy_json/0/evm_npy/hmdb_test_unknown_valid_label.npy")


##################################################################
# UCF101: training
run(root=ucf101_base_dir,
    anno=ucf_101_train_known_json_path,
    batch_size=BS*BS_UPSCALE,
    split='training',
    trained_model_path=ucf_model_path,
    nb_classes=88,
    feature_save_path="/data/jin.huang/ucf101_npy_json/ta2_10_folds/0/evm_npy/ucf_101_train_known_feature.npy",
    label_save_path="/data/jin.huang/ucf101_npy_json/ta2_10_folds/0/evm_npy/ucf_101_train_known_label.npy")

# UCF101: test known feedback
run(root=ucf101_base_dir,
    anno=ucf_101_test_known_feedback_json,
    batch_size=BS*BS_UPSCALE,
    split='feedback_set',
    trained_model_path=ucf_model_path,
    nb_classes=88,
    feature_save_path="/data/jin.huang/ucf101_npy_json/ta2_10_folds/0/evm_npy/ucf_101_test_known_feedback_feature.npy",
    label_save_path="/data/jin.huang/ucf101_npy_json/ta2_10_folds/0/evm_npy/ucf_101_test_known_feedback_label.npy")

# UCF101: test known valid
run(root=ucf101_base_dir,
    anno=ucf_101_test_known_valid_json,
    batch_size=BS*BS_UPSCALE,
    split='test_set',
    trained_model_path=ucf_model_path,
    nb_classes=88,
    feature_save_path="/data/jin.huang/ucf101_npy_json/ta2_10_folds/0/evm_npy/ucf_101_test_known_valid_feature.npy",
    label_save_path="/data/jin.huang/ucf101_npy_json/ta2_10_folds/0/evm_npy/ucf_101_test_known_valid_label.npy")

# UCF101: training
run(root=ucf101_base_dir,
    anno=ucf_101_test_unknown_feedback_json,
    batch_size=BS*BS_UPSCALE,
    split='feedback_set',
    trained_model_path=ucf_model_path,
    nb_classes=88,
    feature_save_path="/data/jin.huang/ucf101_npy_json/ta2_10_folds/0/evm_npy/ucf_101_test_unknown_feedback_feature.npy",
    label_save_path="/data/jin.huang/ucf101_npy_json/ta2_10_folds/0/evm_npy/ucf_101_test_unknown_feedback_label.npy")

# UCF101: training
run(root=ucf101_base_dir,
    anno=ucf_101_test_unknown_valid_json,
    batch_size=BS*BS_UPSCALE,
    split='test_set',
    trained_model_path=ucf_model_path,
    nb_classes=88,
    feature_save_path="/data/jin.huang/ucf101_npy_json/ta2_10_folds/0/evm_npy/ucf_101_test_unknown_valid_feature.npy",
    label_save_path="/data/jin.huang/ucf101_npy_json/ta2_10_folds/0/evm_npy/ucf_101_test_unknown_valid_label.npy")
