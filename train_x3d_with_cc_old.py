# Training X3D with contrastive loss
# Modified based on X3D authors' and Dawei's code and original X3D by Jin Huang

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pkbar
import warnings
import x3d as resnet_x3d
import torchvision
import mlp_network

from utils import contrastive_loss
from utils.apmeter import APMeter
from data.ucf101 import customized_dataset, custom_collate_fn
from utils.transform import Transforms
from transforms.spatial_transforms import Compose, Normalize, \
    RandomHorizontalFlip, MultiScaleRandomCropMultigrid, ToTensor, \
    CenterCropScaled

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()
parser.add_argument('-gpu', default='2', type=str)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu



###############################################################
# TODO: Organize this part for multiple dataset setting...
###############################################################
BS = 8
s=0.5
BS_UPSCALE = 2
# INIT_LR = 0.002 * BS_UPSCALE
INIT_LR = 0.02
GPUS = 1

instance_temperature = 0.5
cluster_temperature = 1.0

X3D_VERSION = 'M'

# This is the root dir to the npy file
# TA2_ROOT = '/data/jin.huang/ucf101_npy_json/'
TA2_ROOT = '/data/jin.huang/hmdb51/npy_json/0'

# This is the path to json file
# TA2_ANNO = '/data/jin.huang/ucf101_npy_json/ucf101.json'
TA2_ANNO = '/data/jin.huang/hmdb51/npy_json/0/ta2_partition_0.json'

# model_save_path = "/data/jin.huang/ucf101_models"
model_save_path = "/data/jin.huang/models/contrastive_clustering/hmdb51/0630_debug"

# TODO: these need to be changed
# TA2_DATASET_SIZE = {'train':13446, 'val':1491}
TA2_DATASET_SIZE = {'train':1906, 'val':212}
# TA2_MEAN = [0, 0, 0]
TA2_MEAN = [0.43216, 0.394666, 0.37645]
# TA2_STD = [1, 1, 1]
TA2_STD = [0.22803, 0.22145, 0.216989]


###############################################################
# TODO: Define the dataloaders here
###############################################################






###############################################################
# TODO: Main function for training and validation
###############################################################

def run(init_lr=INIT_LR,
        max_epochs=100,
        root=TA2_ROOT,
        anno=TA2_ANNO,
        batch_size=BS*BS_UPSCALE):

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
    val_iterations_per_epoch = TA2_DATASET_SIZE['val']//(batch_size)
    max_steps = iterations_per_epoch * max_epochs

    # img_shape = [3, 16, 224, 224]
    # data_augmentation = Transforms(size=img_shape, s=0.5)

    train_spatial_transforms = Compose([MultiScaleRandomCropMultigrid([crop_size/i for i in resize_size], crop_size),
                                        RandomHorizontalFlip(),
                                        torchvision.transforms.RandomApply(
                                            [torchvision.transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)],
                                            p=0.8),
                                        torchvision.transforms.RandomGrayscale(p=0.2),
                                        ToTensor(255),
                                        Normalize(TA2_MEAN, TA2_STD),])

    val_spatial_transforms = Compose([CenterCropScaled(crop_size),
                                        ToTensor(255),
                                        Normalize(TA2_MEAN, TA2_STD)])

    # val_spatial_transforms = train_spatial_transforms

    dataset = customized_dataset(split_file=anno,
                                 split='training',
                                 root=root,
                                 num_classes=26,
                                 spatial_transform=train_spatial_transforms,
                                 frames=80,
                                 gamma_tau=gamma_tau,
                                 crops=1,
                                 use_contrastive_loss=True)

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=4,
                                             pin_memory=True)


    val_dataset = customized_dataset(split_file=anno,
                                     split='validation',
                                     root=root,
                                     num_classes=26,
                                     spatial_transform=val_spatial_transforms,
                                     frames=80,
                                     gamma_tau=gamma_tau,
                                     crops=10,
                                     use_contrastive_loss=True)

    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 num_workers=4,
                                                 pin_memory=True)

    dataloaders = {'train': dataloader, 'val': val_dataloader}
    datasets = {'train': dataset, 'val': val_dataset}
    print('train',len(datasets['train']),'val',len(datasets['val']))
    print('Total iterations:', max_steps, 'Total epochs:', max_epochs)
    print('datasets created')

    x3d = resnet_x3d.generate_model(x3d_version=X3D_VERSION, n_classes=400, n_input_channels=3, dropout=0.5, base_bn_splits=1)
    load_ckpt = torch.load('models/x3d_multigrid_kinetics_fb_pretrained.pt')
    x3d.load_state_dict(load_ckpt['model_state_dict'])
    save_model = model_save_path + '/x3d_ta2_rgb_sgd_'

    # TODO: Create MLP model
    mlp_model = mlp_network.mlp_network(feature_dim=26,
                                        class_num=26).cuda()

    # TODO: change here for different number of class in our dataset
    # x3d.replace_logits(88)
    # x3d.replace_logits(101)
    x3d.replace_logits(26)

    if steps>0:
        load_ckpt = torch.load(model_save_path + '/x3d_ta2_rgb_sgd_'+str(load_steps).zfill(6)+'.pt')
        x3d.load_state_dict(load_ckpt['model_state_dict'])

    x3d = x3d.cuda()
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


    tr_apm_a = APMeter()
    tr_apm_b = APMeter()

    val_apm_a = APMeter()
    val_apm_b = APMeter()

    best_map = 0

    torch.autograd.set_detect_anomaly(True)
    print("Anomaly detection on.")

    while epochs < max_epochs:
        print ('Step {} Epoch {}'.format(steps, epochs))
        print ('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train']+['val']:
        # for phase in ['val']:
            bar_st = iterations_per_epoch if phase == 'train' else val_iterations_per_epoch
            bar = pkbar.Pbar(name='update: ', target=bar_st)

            if phase == 'train':
                x3d.train(True)
                epochs += 1
                torch.autograd.set_grad_enabled(True)
            else:
                x3d.train(False)
                _ = x3d.module.aggregate_sub_bn_stats()
                torch.autograd.set_grad_enabled(False)

            total_loss = 0.0
            # tot_cls_loss = 0.0
            # total_instance_loss = 0.0
            num_iter = 0
            optimizer.zero_grad()

            # Iterate over data.
            # print(phase)
            # print(len(dataloaders["train"]))
            # print(len(dataloaders["val"]))

            for i,data in enumerate(dataloaders[phase]):
                num_iter += 1
                bar.update(i)

                if phase == 'train':
                    # Get 2 augmented input here
                    inputs_a, inputs_b, labels = data

                    inputs_a = inputs_a.cuda() # B 3 T W H
                    inputs_b = inputs_b.cuda()
                    labels = labels.cuda()

                    # Two augmented images sent into a same X3D and gets two feature maps ha and hb
                    # logit shape: [16, 26] ([batch_size, nb_classes])
                    # probs shape: [16, 26] ([batch_size, nb_classes])
                    logits_a, _, _ = x3d(inputs_a)
                    logits_a = logits_a.squeeze(2)

                    logits_b, _, _ = x3d(inputs_b)
                    logits_b = logits_b.squeeze(2)

                    # Send ha and hb into 2 MLP respectively
                    # MLP1: does not have SoftMax, produces feature maps za and zb
                    # MLP2: has a SoftMax, produces soft labels ya and yb
                    # shape of z_a, z_b, y_a, y_b: [batch_size, nb_classes]
                    z_a, z_b, y_a, y_b = mlp_model(logits_a, logits_b)

                    # Unsupervised loss - maximize similarity between za and zb
                    criterion_cluster = contrastive_loss.ClusterLoss(26,
                                                                     cluster_temperature,
                                                                     device=None).cuda()
                    cluster_loss = criterion_cluster(z_a, z_b)

                    # Normal supervised learning for ya and yb
                    instance_loss_a = criterion(y_a, labels)
                    instance_loss_b = criterion(y_b, labels)

                    # TODO: Get training predictions here
                    probs_a = F.sigmoid(y_a)
                    probs_a = torch.max(probs_a, dim=1)[0]

                    probs_b = F.sigmoid(y_b)
                    probs_b = torch.max(probs_b, dim=1)[0]

                    # TODO: Get mAP
                    tr_apm_a.add(probs_a.detach().cpu().numpy(), labels.cpu().numpy())
                    tr_apm_b.add(probs_b.detach().cpu().numpy(), labels.cpu().numpy())

                    # Calculate total loss
                    loss =  instance_loss_a + instance_loss_b - cluster_loss
                    loss.backward()

                # validation process here
                else:
                    # Get 2 augmented input here
                    inputs_a, inputs_b, labels = data

                    inputs_a = inputs_a.cuda()  # B 3 T W H
                    inputs_b = inputs_b.cuda()
                    labels = labels.cuda()

                    # TODO: check and fix input shape
                    # print(len(inputs_a.shape))
                    # print("\n")
                    # print("@" * 20)
                    # print(i)
                    # print("@" * 20)

                    if len(inputs_a.shape) != 5:
                        inputs_a = torch.squeeze(inputs_a)
                        # print(inputs_a.shape)
                    if len(inputs_b.shape) != 5:
                        inputs_b = torch.squeeze(inputs_b)
                        # print(inputs_b.shape)
                    #
                    # print("\n")
                    # print("@" * 20)
                    # print(inputs_a.shape)
                    # print(inputs_b.shape)
                    # print("@" * 20)

                    # Two augmented images sent into a same X3D and gets two feature maps
                    logits_a, _, _ = x3d(inputs_a)
                    logits_a = logits_a.squeeze(2)

                    logits_b, _, _ = x3d(inputs_b)
                    logits_b = logits_b.squeeze(2)

                    # Send ha and hb into 2 MLP respectively
                    z_a, z_b, y_a, y_b = mlp_model(logits_a, logits_b)

                    # Unsupervised loss - maximize similarity between za and zb
                    criterion_cluster = contrastive_loss.ClusterLoss(26,
                                                                     cluster_temperature,
                                                                     device=None).cuda()
                    cluster_loss = criterion_cluster(z_a, z_b)

                    # Normal supervised learning for ya and yb
                    instance_loss_a = criterion(y_a, labels)
                    instance_loss_b = criterion(y_b, labels)

                    loss = instance_loss_a + instance_loss_b - cluster_loss

                    # Probabilities
                    probs_a = F.sigmoid(y_a)
                    probs_a = torch.max(probs_a, dim=1)[0]

                    probs_b = F.sigmoid(y_b)
                    probs_b = torch.max(probs_b, dim=1)[0]

                    # mAP
                    val_apm_a.add(probs_a.detach().cpu().numpy(), labels.cpu().numpy())
                    val_apm_b.add(probs_b.detach().cpu().numpy(), labels.cpu().numpy())

                total_loss += loss.item()

            # if (num_iter == num_steps_per_update) and (phase == "train"):
            if phase == "train":
                steps += 1
                num_iter = 0
                optimizer.step()
                optimizer.zero_grad()
                s_times = iterations_per_epoch // 2

                if (steps - load_steps) % s_times == 0:
                    tr_map_a = tr_apm_a.value().mean()
                    tr_map_b = tr_apm_a.value().mean()

                    tr_map_a.reset()
                    tr_map_b.reset()

                    print('Epoch (training):{} '
                          'Cls Loss: {:.4f} '
                          'mAP branch a: {:.4f} '
                          'mAP branch a: {:.4f} '.format(epochs,
                                                         total_loss,
                                                         tr_map_a,
                                                         tr_map_b))

            if phase == 'val':
                val_map_a = val_apm_a.value().mean()
                val_map_b = val_apm_b.value().mean()

                lr_sched.step(total_loss)

                val_apm_a.reset()
                val_apm_b.reset()

                print (' Epoch (validation):{}  '
                       ' Validation Loss: {:.4f} '
                       ' mAP (branch A): {:.4f} '
                       ' mAP (branch B): {:.4f}\n'.format(epochs,
                                                          total_loss,
                                                          val_map_a,
                                                          val_map_b))

                total_loss = 0.0

                if (val_map_a > best_map) and (val_map_b > best_map):
                    ckpt = {'model_state_dict': x3d.module.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': lr_sched.state_dict()}

                    if val_map_a > val_map_b:
                        best_map = val_map_a
                    else:
                        best_map = val_map_b

                    best_epoch = epochs

                    print (' Epoch:{} {} best mAP: {:.4f}\n'.format(best_epoch, phase, best_map))
                    torch.save(ckpt, save_model+'best.pt')




def lr_warmup(init_lr, cur_steps, warmup_steps, opt):
    start_after = 1
    if cur_steps < warmup_steps and cur_steps > start_after:
        lr_scale = min(1., float(cur_steps + 1) / warmup_steps)
        for pg in opt.param_groups:
            pg['lr'] = lr_scale * init_lr




def print_stats(long_ind, batch_size, stats, gamma_tau, bn_splits, lr):
    bs = batch_size * LONG_CYCLE[long_ind]
    if long_ind in [0,1]:
        bs = [bs*j for j in [2,1]]
        print(' ***** LR {} Frames {}/{} BS ({},{}) W/H ({},{}) BN_splits {} long_ind {} *****'.format(lr, stats[0][0], gamma_tau, bs[0], bs[1], stats[2][0], stats[3][0], bn_splits, long_ind))
    else:
        bs = [bs*j for j in [4,2,1]]
        print(' ***** LR {} Frames {}/{} BS ({},{},{}) W/H ({},{},{}) BN_splits {} long_ind {} *****'.format(lr, stats[0][0], gamma_tau, bs[0], bs[1], bs[2], stats[1][0], stats[2][0], stats[3][0], bn_splits, long_ind))


if __name__ == '__main__':
    run()