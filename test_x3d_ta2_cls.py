import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms
from torchsummary import summary

import numpy as np
from barbar import Bar
import pkbar
from apmeter import APMeter

import x3d as resnet_x3d

from ucf101 import UCF101

from transforms.spatial_transforms import Compose, Normalize, RandomHorizontalFlip, MultiScaleRandomCrop, MultiScaleRandomCropMultigrid, ToTensor, CenterCrop, CenterCropScaled
from transforms.temporal_transforms import TemporalRandomCrop
from transforms.target_transforms import ClassLabel
import pdb

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('-gpu', default='0,1', type=str)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu


BS = 16
BS_UPSCALE = 2
GPUS = 2

X3D_VERSION = 'M'

TA2_ROOT = '../datasets/TA2_splits'
TA2_ANNO = '../datasets/TA2_splits/TA2_classification.json'
TA2_DATASET_SIZE = {'train':13446, 'val':1491}
TA2_MEAN = [0, 0, 0]
TA2_STD = [1, 1, 1]

# warmup_steps=0
def run(root=TA2_ROOT, anno=TA2_ANNO, batch_size=BS*BS_UPSCALE):

    frames=80 # DOUBLED INSIDE DATASET, AS LONGER CLIPS
    crop_size = {'S':160, 'M':224, 'XL':312}[X3D_VERSION]
    resize_size = {'S':[180.,225.], 'M':[256.,256.], 'XL':[360.,450.]}[X3D_VERSION] # 'M':[256.,320.] FOR LONGER SCHEDULE
    gamma_tau = {'S':6, 'M':5, 'XL':5}[X3D_VERSION] # DOUBLED INSIDE DATASET, AS LONGER CLIPS

    val_iterations_per_epoch = TA2_DATASET_SIZE['val']//(batch_size)
    val_spatial_transforms = Compose([CenterCropScaled(crop_size),
                                        ToTensor(255),
                                        Normalize(TA2_MEAN, TA2_STD)])

    val_dataset = UCF101(anno, 'validation', root, val_spatial_transforms, frames=80, gamma_tau=gamma_tau, crops=10)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                                num_workers=8, pin_memory=True)

    x3d = resnet_x3d.generate_model(x3d_version=X3D_VERSION, n_classes=88, n_input_channels=3, dropout=0.5, base_bn_splits=1)
    load_ckpt = torch.load('models/x3d_ta2_rgb_sgd_best.pt')
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
    _ = x3d.module.aggregate_sub_bn_stats() # FOR EVAL AGGREGATE BN STATS

    # Iterate over data.
    for i,data in enumerate(val_dataloader):
        bar.update(i)
        inputs, labels = data
        b,n,c,t,h,w = inputs.shape # FOR MULTIPLE TEMPORAL CROPS
        inputs = inputs.view(b*n,c,t,h,w)

        inputs = inputs.cuda() # B 3 T W H
        labels = labels.cuda() # B C
        with torch.no_grad():
            logits, feat, base = x3d(inputs)
        logits = logits.squeeze(2) # B C
        logits = logits.view(b,n,logits.shape[1]) # FOR MULTIPLE TEMPORAL CROPS
        pdb.set_trace()
        probs = F.sigmoid(logits)
        #probs = torch.mean(probs, 1)
        #logits = torch.mean(logits, 1)
        probs = torch.max(probs, dim=1)[0]
        logits = torch.max(logits, dim=1)[0]

        val_apm.add(probs.detach().cpu().numpy(), labels.cpu().numpy())

    val_map = val_apm.value().mean()
    print ('Epoch:{} val mAP: {:.4f}'.format(epochs, val_map))


if __name__ == '__main__':
    run()
