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
from data.ucf101 import customized_dataset
from utils.transform import Transforms
from transforms.spatial_transforms import Compose, Normalize, \
    RandomHorizontalFlip, MultiScaleRandomCropMultigrid, ToTensor, \
    CenterCropScaled

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()
parser.add_argument('-gpu', default='0', type=str)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu



###############################################################
# TODO: Organize this part for multiple dataset setting...
###############################################################
BS = 8
s=0.5
BS_UPSCALE = 2
INIT_LR = 0.02 * BS_UPSCALE
GPUS = 2

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
model_save_path = "/data/jin.huang/hmdb51_models/0614_x3d_p0"

# TODO: these need to be changed
# TA2_DATASET_SIZE = {'train':13446, 'val':1491}
TA2_DATASET_SIZE = {'train':1906, 'val':212}
TA2_MEAN = [0, 0, 0]
TA2_STD = [1, 1, 1]


###############################################################
# TODO: Define the dataloaders here
###############################################################






###############################################################
# TODO: Main function for training and validation
###############################################################

def run(init_lr=INIT_LR,
        max_epochs=1,
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
    val_iterations_per_epoch = TA2_DATASET_SIZE['val']//(batch_size//2)
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

    dataset = customized_dataset(split_file=anno,
                                 split='training',
                                 root=root,
                                 num_classes=26,
                                 spatial_transform=train_spatial_transforms,
                                 frames=80,
                                 gamma_tau=gamma_tau,
                                 crops=1)

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
                                     crops=10)
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
    save_model = model_save_path + '/x3d_ta2_rgb_sgd_'

    # TODO: Create MLP model
    mlp_model = mlp_network.mlp_network(feature_dim=26,
                                        class_num=26)

    # TODO: change here for different number of class in our dataset
    # x3d.replace_logits(88)
    # x3d.replace_logits(101)
    x3d.replace_logits(26)

    if steps>0:
        load_ckpt = torch.load(model_save_path + '/x3d_ta2_rgb_sgd_'+str(load_steps).zfill(6)+'.pt')
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
    tr_apm_a = APMeter()
    tr_apm_b = APMeter()

    best_map = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
                x3d.train(False)
                _ = x3d.module.aggregate_sub_bn_stats()
                torch.autograd.set_grad_enabled(False)

            tot_loss = 0.0
            tot_cls_loss = 0.0
            total_instance_loss = 0.0
            num_iter = 0
            optimizer.zero_grad()

            # Iterate over data.
            print(phase)
            for i,data in enumerate(dataloaders[phase]):
                num_iter += 1
                bar.update(i)
                if phase == 'train':
                    # TODO: Get 2 augmented input here
                    inputs_a, inputs_b, labels = data

                    inputs_a = inputs_a.to(device)  # B 3 T W H
                    inputs_b = inputs_b.to(device)
                    labels = labels.to(device)  # B C

                    # TODO: Two augmented images sent into a same X3D
                    #  and gets two feature maps ha and hb
                    logits_a, _, _ = x3d(inputs_a)
                    logits_a = logits_a.squeeze(2)
                    # probs = F.sigmoid(logits_a)

                    logits_b, _, _ = x3d(inputs_b)
                    logits_b = logits_b.squeeze(2)
                    # probs_b = F.sigmoid(logits_b)

                    # print(logits_a.shape)
                    # print(probs.shape)

                    # logit shape: [16, 26] ([batch_size, nb_classes])
                    # probs shape: [16, 26] ([batch_size, nb_classes])

                    # TODO: Send ha and hb into 2 MLP respectively
                    # TODO: MLP1: does not have SoftMax, produces feature maps za and zb
                    # TODO: MLP2: has a SoftMax, produces soft labels ya and yb
                    z_a, z_b, y_a, y_b = mlp_model(logits_a, logits_b)
                    # shape of z_a, z_b, y_a, y_b: [batch_size, nb_classes]


                    # loss_device = torch.device("cuda")
                    # loss_device = torch.cuda.set_device(1)

                    # TODO: Unsupervised loss - maximize similarity between za and zb
                    criterion_cluster = contrastive_loss.ClusterLoss(26,
                                                                     cluster_temperature,
                                                                     device).to(device)
                    cluster_loss = criterion_cluster(z_a, z_b)

                    # # TODO: Normal supervised learning for ya and yb
                    # # TODO(Q): is it necessary to maximize similarity for ya and yb too?
                    criterion_instance = contrastive_loss.InstanceLoss(batch_size,
                                                                       instance_temperature,
                                                                       device).to(device)
                    instance_loss = criterion_instance(y_a, y_b, labels)

                    # ya and yb are logits (modified)
                    # instance_loss_a = criterion(y_a.to(loss_device), labels)
                    # instance_loss_b = criterion(y_b.to(loss_device), labels)

                    # z_a = z_a.to(device)
                    # z_b.to(device)
                    # y_a.to(device)
                    # y_b.to(device)

                    instance_loss_a = criterion(y_a, labels)
                    instance_loss_b = criterion(y_b, labels)

                    total_instance_loss += instance_loss_a.item()
                    total_instance_loss += instance_loss_b.item()

                    loss = cluster_loss + instance_loss_a + instance_loss_b

                    probs_a = y_a
                    probs_b = y_b

                    tr_apm_a.add(probs_a.detach().cpu().numpy(), labels.cpu().numpy())
                    tr_apm_b.add(probs_b.detach().cpu().numpy(), labels.cpu().numpy())

                    loss.backward()

                # TODO: fix the validation phase later
                else:
                    inputs, labels = data
                    b,n,c,t,h,w = inputs.shape # FOR MULTIPLE TEMPORAL CROPS
                    inputs = inputs.view(b*n,c,t,h,w)

                    inputs = inputs.cuda() # B 3 T W H
                    labels = labels.cuda() # B C

                    with torch.no_grad():
                        logits, _, _ = x3d(inputs)
                    logits = logits.squeeze(2) # B C
                    logits = logits.view(b,n,logits.shape[1]) # FOR MULTIPLE TEMPORAL CROPS
                    probs = F.sigmoid(logits)
                    probs = torch.max(probs, dim=1)[0]
                    logits = torch.max(logits, dim=1)[0]

                # cls_loss = criterion(logits, labels)
                # tot_cls_loss += cls_loss.item()
                #
                # loss = cls_loss / num_steps_per_update
                # tot_loss += loss.item()

                if phase == 'train':
                    # tr_apm.add(probs.detach().cpu().numpy(), labels.cpu().numpy())
                    # tr_apm_a.add(probs_a.detach().cpu().numpy(), labels.cpu().numpy())
                    # tr_apm_b.add(probs_b.detach().cpu().numpy(), labels.cpu().numpy())
                    # loss.backward()
                    pass
                else:
                    val_apm.add(probs.detach().cpu().numpy(), labels.cpu().numpy())


                if num_iter == num_steps_per_update and phase == 'train':
                    #lr_warmup(lr, steps-st_steps, warmup_steps, optimizer)
                    steps += 1
                    num_iter = 0
                    optimizer.step()
                    optimizer.zero_grad()
                    s_times = iterations_per_epoch//2
                    if (steps-load_steps) % s_times == 0:
                        tr_map_a = tr_apm_a.value().mean()
                        tr_map_b = tr_apm_a.value().mean()

                        tr_map_a.reset()
                        tr_map_b.reset()

                        print('Epoch (training):{} '
                              'Cls Loss: {:.4f} '
                              'mAP branch a: {:.4f} '
                              'mAP branch a: {:.4f} '.format(epochs,
                                                      loss,
                                                      tr_map_a,
                                                      tr_map_b))

                        # print (' Epoch:{} {} steps: {} Cls Loss: {:.4f} Tot Loss: {:.4f} mAP: {:.4f}\n'.format(epochs, phase,
                        #     steps, tot_cls_loss/(s_times*num_steps_per_update), tot_loss/s_times, tr_map))#, tot_acc/(s_times*num_steps_per_update)))
                        # tot_loss = tot_cls_loss = 0.
                    '''if steps % (1000) == 0:
                        ckpt = {'model_state_dict': x3d.module.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'scheduler_state_dict': lr_sched.state_dict()}
                        torch.save(ckpt, save_model+str(steps).zfill(6)+'.pt')'''

            if phase == 'val':
                pass
                # val_map = val_apm.value().mean()
                # lr_sched.step(tot_loss)
                # val_apm.reset()
                # print (' Epoch:{} {} Loc Cls Loss: {:.4f} Tot Loss: {:.4f} mAP: {:.4f}\n'.format(epochs, phase,
                #     tot_cls_loss/num_iter, (tot_loss*num_steps_per_update)/num_iter, val_map))
                # tot_loss = tot_cls_loss = 0.
                # if val_map > best_map:
                #     ckpt = {'model_state_dict': x3d.module.state_dict(),
                #             'optimizer_state_dict': optimizer.state_dict(),
                #             'scheduler_state_dict': lr_sched.state_dict()}
                #     best_map = val_map
                #     best_epoch = epochs
                #     print (' Epoch:{} {} best mAP: {:.4f}\n'.format(best_epoch, phase, best_map))
                #     torch.save(ckpt, save_model+'best.pt')

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