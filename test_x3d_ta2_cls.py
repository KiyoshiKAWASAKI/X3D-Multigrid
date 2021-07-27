import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

import pkbar
from utils.apmeter import APMeter

import x3d as resnet_x3d

from data.ucf101 import UCF101
import torch.optim as optim

from transforms.spatial_transforms import Compose, Normalize, ToTensor, \
    CenterCropScaled
import sys
import numpy as np

from transforms.spatial_transforms_old import Compose, Normalize, \
    RandomHorizontalFlip, MultiScaleRandomCrop, \
    MultiScaleRandomCropMultigrid, ToTensor, \
    CenterCrop, CenterCropScaled

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('-gpu', default='0', type=str)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu


TA2_MEAN = [0, 0, 0]
TA2_STD = [1, 1, 1]

BS = 2
BS_UPSCALE = 2
INIT_LR = 0.005 * BS_UPSCALE
GPUS = 1
X3D_VERSION = 'M'

###############################
dataset_used = "hmdb"
test_known = True
use_feedback = True
threshold = 0.6

update_fre = 4
#################################
# TODO ()
if dataset_used == "ucf101":
    pass
    # TA2_ROOT = '/data/jin.huang/ucf101_npy_json/ta2_10_folds/0'
    # trained_model_path = "/data/jin.huang/models/x3d/thresholding/0702_ucf/x3d_ta2_rgb_sgd_best.pt"
    # nb_classes = 88
    #
    # if test_known == True:
    #     if use_feedback == False:
    #         TA2_ANNO = '/data/jin.huang/ucf101_npy_json/ta2_10_folds/0/ta2_partition_0_test_known_test.json'
    #         TA2_DATASET_SIZE = {'train': None, 'val': 1026}
    #     else:
    #         TA2_ANNO = '/data/jin.huang/ucf101_npy_json/ta2_10_folds/0/ta2_partition_0_test_known_test.json'
    #         TA2_FEEDBACK = '/data/jin.huang/ucf101_npy_json/ta2_10_folds/0/ta2_partition_0_test_known_feedback.json'
    #         TA2_DATASET_SIZE = {'train': None, 'val': 1026}
    #
    # else:
    #     if use_feedback == False:
    #         TA2_ANNO = '/data/jin.huang/ucf101_npy_json/ta2_10_folds/0/ta2_partition_0_test_unknown_test.json'
    #         TA2_DATASET_SIZE = {'train': None, 'val': 1026}
    #     else:
    #         TA2_ANNO = '/data/jin.huang/ucf101_npy_json/ta2_10_folds/0/ta2_partition_0_test_unknown_test.json'
    #         TA2_FEEDBACK = '/data/jin.huang/ucf101_npy_json/ta2_10_folds/0/ta2_partition_0_test_unknown_feedback.json'
    #         TA2_DATASET_SIZE = {'train': None, 'val': 1026}

else:
    trainining_json_path = "/data/jin.huang/hmdb51/npy_json/0/ta2_partition_0.json"
    TA2_ROOT = "/data/jin.huang/hmdb51/npy_json/0"
    trained_model_path = "/data/jin.huang/models/x3d/thresholding/0702_hmdb/x3d_ta2_rgb_sgd_best.pt"
    nb_classes = 26

    if test_known == True:
        if use_feedback == False:
            TA2_ANNO = "/data/jin.huang/hmdb51/npy_json/0/ta2_partition_0_test_known_test.json"
            TA2_DATASET_SIZE = {'train': None, 'val': 464}
        else:
            TA2_ANNO = "/data/jin.huang/hmdb51/npy_json/0/ta2_partition_0_test_known_test.json"
            TA2_FEEDBACK = "/data/jin.huang/hmdb51/npy_json/0/ta2_partition_0_test_known_feedback.json"
            TA2_DATASET_SIZE = {'train': None, 'val': 464}
    else:
        if use_feedback == False:
            TA2_ANNO = "/data/jin.huang/hmdb51/npy_json/0/ta2_partition_0_test_unknown_test.json"
            TA2_DATASET_SIZE = {'train': None, 'val': 464}
        else:
            TA2_ANNO = "/data/jin.huang/hmdb51/npy_json/0/ta2_partition_0_test_unknown_test.json"
            TA2_FEEDBACK = "/data/jin.huang/hmdb51/npy_json/0/ta2_partition_0_test_unknown_test.json"
            TA2_DATASET_SIZE = {'train': None, 'val': 464}


def no_requires_grad(model, fixed_layer_names):
    for layer_name in fixed_layer_names:
        if hasattr(model, layer_name):
            getattr(model, layer_name).requires_grad = False
    else:
        print(f"Trying to set non existent layer: {layer_name}")



def run(root=TA2_ROOT, anno=TA2_ANNO,batch_size=BS*BS_UPSCALE):

    frames=80 # DOUBLED INSIDE DATASET, AS LONGER CLIPS
    crop_size = {'S':160, 'M':224, 'XL':312}[X3D_VERSION]
    resize_size = {'S':[180.,225.], 'M':[256.,256.], 'XL':[360.,450.]}[X3D_VERSION] # 'M':[256.,320.] FOR LONGER SCHEDULE
    gamma_tau = {'S':6, 'M':5, 'XL':5}[X3D_VERSION] # DOUBLED INSIDE DATASET, AS LONGER CLIPS


    val_spatial_transforms = Compose([CenterCropScaled(crop_size),
                                        ToTensor(255),
                                        Normalize(TA2_MEAN, TA2_STD)])


    val_dataset = UCF101(split_file=anno,
                         split='test_set',
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
                                                 num_workers=4,
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

    val_apm = APMeter()
    epochs = load_ckpt['scheduler_state_dict']['last_epoch']

    print ('-' * 10)
    bar_st = val_iterations_per_epoch
    bar = pkbar.Pbar(name='evaluating: ', target=bar_st)
    x3d.train(False)  # Set model to evaluate mode
    _ = x3d.module.aggregate_sub_bn_stats() # FOR EVAL AGGREGATE BN STATS\

    if test_known == False:
        print("Testing unknown. Threshold %f" % threshold)
        count_correct = 0
        count_wrong = 0
        sm = torch.nn.Softmax(dim=1)


    # Iterate over data.
    # for i in range(len(val_dataloader)):
    for i, data in enumerate(val_dataloader):
        # bar.update(i)
        # print(i)
        # try:
        inputs, labels = next(iter(val_dataloader))
            # print(inputs.shape)
        # except Exception as e:
        #     print(e)
        #     continue

        b,n,c,t,h,w = inputs.shape # FOR MULTIPLE TEMPORAL CROPS
        inputs = inputs.view(b*n,c,t,h,w)

        # b, c, t, h, w = inputs.shape  # FOR MULTIPLE TEMPORAL CROPS
        # inputs = inputs.view(b , c, t, h, w)

        inputs = inputs.cuda() # B 3 T W H
        labels = labels.cuda() # B C

        with torch.no_grad():
            logits, feat, base = x3d(inputs)

        logits = logits.squeeze(2) # B C
        logits = logits.view(b,n,logits.shape[1]) # FOR MULTIPLE TEMPORAL CROPS
        # logits = logits.view(b, logits.shape[1]) # FOR MULTIPLE TEMPORAL CROPS
        # pdb.set_trace()
        probs = F.sigmoid(logits)
        #probs = torch.mean(probs, 1)
        #logits = torch.mean(logits, 1)
        probs = torch.max(probs, dim=1)[0]
        logits = torch.max(logits, dim=1)[0]

        # Shape of probs: [batch_size, nb_classes]

        if test_known == False:
            # TODO: Get max probability for each sample
            probs_sm = sm(probs)

            for one_prob in probs_sm:
                max_prob = torch.max(one_prob)

                if max_prob > threshold:
                    count_wrong += 1
                else:
                    count_correct += 1

            print("Update - Correct: %d. Wrong: %d" % (count_correct, count_wrong))

        if test_known:
            val_apm.add(probs.detach().cpu().numpy(), labels.cpu().numpy())

    if test_known:
        val_map = val_apm.value().mean()
        print ('Epoch:{} val mAP: {:.4f}'.format(epochs, val_map))
    else:
        print("Total unknown samples: %d" % (count_correct + count_wrong))
        print("Unknown as unknown: %d" % count_correct)
        print("Unknown as known: %d" % count_wrong)
        print("Accuracy: %f" % (float(count_correct)/(float(count_correct+count_wrong))))




def run_with_feedback(root=TA2_ROOT,
                      anno=TA2_ANNO,
                      feedback_file=TA2_FEEDBACK,
                      batch_size=BS*BS_UPSCALE,
                      init_lr=INIT_LR):

    frames=80 # DOUBLED INSIDE DATASET, AS LONGER CLIPS
    crop_size = {'S':160, 'M':224, 'XL':312}[X3D_VERSION]
    resize_size = {'S':[180.,225.], 'M':[256.,256.], 'XL':[360.,450.]}[X3D_VERSION] # 'M':[256.,320.] FOR LONGER SCHEDULE
    gamma_tau = {'S':6, 'M':5, 'XL':5}[X3D_VERSION] # DOUBLED INSIDE DATASET, AS LONGER CLIPS

    train_spatial_transforms = Compose([MultiScaleRandomCropMultigrid([crop_size / i for i in resize_size], crop_size),
                                        RandomHorizontalFlip(),
                                        ToTensor(255),
                                        Normalize(TA2_MEAN, TA2_STD)])


    val_spatial_transforms = Compose([CenterCropScaled(crop_size),
                                        ToTensor(255),
                                        Normalize(TA2_MEAN, TA2_STD)])

    # TODO: Here is the dataloader for re-using the training data
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



    # TODO: here is the dataloader for testing
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
                                                 num_workers=0,
                                                 pin_memory=True)

    # TODO: here is the dataloader for getting feedback data
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
                                                 num_workers=0,
                                                 pin_memory=True)

    x3d = resnet_x3d.generate_model(x3d_version=X3D_VERSION,
                                    n_classes=nb_classes,
                                    n_input_channels=3,
                                    dropout=0.5,
                                    base_bn_splits=1)

    # TODO: need to update this
    val_iterations_per_epoch = TA2_DATASET_SIZE['val'] // (batch_size)

    load_ckpt = torch.load(trained_model_path, map_location='cpu')
    x3d.load_state_dict(load_ckpt['model_state_dict'])

    x3d.cuda()

    # TODO: freeze some layers
    for name, param in x3d.named_parameters():
        if param.requires_grad:
            print(name, param.data)
    # count = 0
    # for parameter in x3d.parameters():
    #     if count != 316:
    #         parameter.requires_grad = False
    #         count += 1
    #     else:
    #         parameter.requires_grad = True

    x3d = nn.DataParallel(x3d)
    print('model loaded')

    val_apm = APMeter()
    feedback_apm = APMeter()
    epochs = load_ckpt['scheduler_state_dict']['last_epoch']

    print ('-' * 10)
    bar_st = val_iterations_per_epoch
    bar = pkbar.Pbar(name='Fine-tuning using feedback ', target=bar_st)
    # x3d.train(False)  # Set model to evaluate mode
    # _ = x3d.module.aggregate_sub_bn_stats() # FOR EVAL AGGREGATE BN STATS

    lr = init_lr
    print('INIT LR: %f' % lr)

    optimizer = optim.SGD(x3d.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5)
    # lr_sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.1, verbose=True)

    criterion = nn.BCEWithLogitsLoss()
    x3d.train(True)
    torch.autograd.set_grad_enabled(True)
    optimizer.zero_grad()


    # Iterate over data.
    total_loss = 0

    # TODO: Use feedback set for updating and fine-tuning
    # TODO: first step, only using knowns
    train_dataloader_iterator = iter(train_dataloader)

    for i, data in enumerate(feedback_dataloader):
        data_feedback, labels_feedback = data

        try:
            data_train, labels_train = next(train_dataloader_iterator)
        except StopIteration:
            train_dataloader_iterator = iter(train_dataloader)
            data_train, labels_train = next(train_dataloader_iterator)

        # TODO: Just update the network every 4 steps
        if (i == 3) or (i % 4 == 3):
            # Get data from training set and feedback set
            b_fb, n_fb, c_fb, t_fb, h_fb, w_fb = data_feedback.shape
            data_feedback = data_feedback.view(b_fb * n_fb, c_fb, t_fb, h_fb, w_fb)

            b_train, n_train, c_train, t_train, h_train, w_train = data_train.shape
            data_train = data_train.view(b_train * n_train, c_train, t_train, h_train, w_train)

            # Stack the input and use it to update the network
            inputs = torch.cat((data_feedback, data_train), dim=0)
            labels = torch.cat((labels_feedback, labels_train), dim=0)

            inputs = inputs.cuda()
            labels = labels.cuda()

            print("input shape:", inputs.shape)
            print("label shape", labels.shape)

            # TODO: Updating network
            logits, feat, base = x3d(inputs)

            logits = logits.squeeze(2) # B C
            logits = logits.view(b_fb*2, n_fb, logits.shape[1])
            logits = torch.max(logits, dim=1)[0]
            probs = F.sigmoid(logits)


            loss = criterion(logits, labels)
            total_loss += loss.item()
            loss.requres_grad = True
            loss.backward()

            feedback_apm.add(probs.detach().cpu().numpy(), labels.cpu().numpy())

            # TODO: Do test here
            if test_known == False:
                pass

            else:
                for k, data in enumerate(val_dataloader):
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
                print('Epoch (testing after updating the network):{} val mAP: {:.4f}'.format(epochs, val_map))






        else:
            continue




    # feedback_amp = feedback_apm.value().mean()
    # print ('Epoch:{} val mAP: {:.4f}'.format(epochs, feedback_amp))


if __name__ == '__main__':
    if use_feedback == False:
        run()

    else:
        run_with_feedback()
        # pass