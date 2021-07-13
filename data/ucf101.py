import torch
import torch.utils.data as data_utl
from torch.utils.data.dataloader import default_collate

import numpy as np
import json
import csv
import h5py
import random
import os
import os.path
import functools

import torchvision
from PIL import Image
import pdb
import cv2
from utils.transform import Transforms


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def get_default_image_loader():
    torchvision.set_image_backend('accimage')
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader


def video_loader(video_dir_path, vid, frame_indices, image_loader):
    video = []
    for i in frame_indices:
        image_path = os.path.join(video_dir_path, vid, vid+'-'+str(i).zfill(6)+'.jpg')
        if os.path.exists(image_path):
            video.append(image_loader(image_path))
        else:
            return video

    return video


def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)


def my_video_loader(seq_path, frame_indices):
    frames = []
    # extract frames from the video
    if os.path.exists(seq_path):
        cap = cv2.VideoCapture(seq_path)
        cnt = 0
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == False:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # convert opencv image to PIL
            frames.append(frame)
        test = min(len(frames)-1, max(frame_indices)-1)
        #print(seq_path, len(frames), min(frame_indices), max(frame_indices), test)
        clip_frames = [frames[min(len(frames)-1, cnt-1)] for cnt in frame_indices]

    else:
        print('{} does not exist'.format(seq_path))

    return clip_frames

def get_class_labels(data):
    class_labels_map = {}
    index = 0

    # print(data["labels"])

    for class_label in data['labels']:
        class_labels_map[class_label] = index
        index += 1

    # print(class_labels_map)
    return class_labels_map

def get_video_names_and_annotations(data, subset):
    video_names = []
    annotations = []
    # print(data['database'].items()) # Correct

    for key, value in data['database'].items():
        this_subset = value['subset']
        if this_subset == subset:
            # label = value['annotations']['label']
            video_names.append(key)
            annotations.append(value['annotations'])
    return video_names, annotations


def load_rgb_frames(image_dir, vid, start, num, stride, video_loader):
    frame_indices = list(range(start, start+num, stride))
    frames = my_video_loader(vid, frame_indices)

    return frames

# def make_dataset(split_file, split, root, num_classes=101):
def make_dataset(split_file, split, root, num_classes=26):
    dataset = []
    with open(split_file, 'r') as f:
        data = json.load(f)
    video_names, annotations = get_video_names_and_annotations(data, split)
    class_to_idx = get_class_labels(data)

    print(len(video_names))

    idx_to_class = {}
    for name, label in class_to_idx.items():
        idx_to_class[label] = name

    pre_data_file = split_file[:-5]+'_'+split+'_ta2.npy'
    if os.path.exists(pre_data_file):
        print('{} exists'.format(pre_data_file))
        dataset = np.load(pre_data_file, allow_pickle=True)
    else:
        print('{} does not exist'.format(pre_data_file))
        for i in range(len(video_names)):
            if i % 1000 == 0:
                print('Loading videos [{}/{}]'.format(i, len(video_names)))
            video_path = video_names[i]
            if not os.path.exists(video_path):
                continue

            num_frames = int(data['database'][video_names[i]]['annotations']['segment'][1])
            if num_frames > 0:
                num_frames = max(2*80+2, num_frames)
            else:
                continue

            label = np.zeros((num_classes,num_frames), np.float32)
            cur_class_idx = class_to_idx[annotations[i]['label']]
            label[cur_class_idx, :] = 1
            dataset.append((video_path, label, num_frames))
        np.save(pre_data_file, dataset)

    print('dataset size:%d'%len(dataset))

    return dataset


class UCF101(data_utl.Dataset):

    def __init__(self, split_file, split,
                 root, num_classes=88, spatial_transform=None,
                 task='class', frames=80, gamma_tau=5, crops=1,
                 test_phase=False, is_feedback=False):
    # def __init__(self, split_file, split, root, num_classes=101, spatial_transform=None, task='class', frames=80, gamma_tau=5, crops=1):

        self.data = make_dataset(split_file, split, root, num_classes)
        self.split_file = split_file
        self.root = root
        self.frames = frames * 2
        self.gamma_tau = gamma_tau * 2
        self.loader = get_default_video_loader()
        self.spatial_transform = spatial_transform
        self.crops = crops
        self.split = split
        self.task = task
        self.test_phase = test_phase
        self.is_feedback = is_feedback


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        if self.test_phase:
            if self.is_feedback:
                self.split = "feedback_set"
            else:
                self.split = "test_set"

        else:
            if self.crops > 1:
                self.split = 'validation'

        vid, label, nf = self.data[index]

        if (self.split == "validation") or \
            (self.split == "feedback_set") or \
            (self.split == "test_set"):
            frames = nf
            start_f = 1
        else:
            frames = self.frames
            start_f = random.randint(1,nf-(self.frames+1))

        stride_f = self.gamma_tau
        imgs = load_rgb_frames(self.root, vid, start_f, frames, stride_f, self.loader)
        label = label[:, start_f-1:start_f-1+frames:1] #stride_f
        label = torch.from_numpy(label)
        if self.task == 'class':
            label = torch.max(label, dim=1)[0] # C T --> C
        if self.spatial_transform is not None:
            # print(self.spatial_transform )
            self.spatial_transform.randomize_parameters(224)
            imgs_l = [self.spatial_transform(Image.fromarray(img)) for img in imgs]
        imgs_l = torch.stack(imgs_l, 0).permute(1, 0, 2, 3) # T C H W --> C T H W

        if ((self.split=='validation') or
            (self.split=="feedback_set") or
            (self.split=="test_set") ) \
                and self.task == 'class': #self.crops > 1:
            step = int((imgs_l.shape[1] - 1 - self.frames//self.gamma_tau)//(self.crops-1))
            if step == 0:
                clips = [imgs_l[:,:self.frames//self.gamma_tau,...] for i in range(self.crops)]
                clips = torch.stack(clips, 0)
                # print("clips when step == 0")
                # print(clips.shape)
            else:
                clips = [imgs_l[:,i:i+self.frames//self.gamma_tau,...] for i in range(0, step*self.crops, step)]
                clips = torch.stack(clips, 0)
                # print("clips when step != 0")
                # print(clips.shape)
        else:
            clips = imgs_l

        # print("@" * 20)

        return clips, label

    def __len__(self):
        return len(self.data)


class customized_dataset(data_utl.Dataset):

    def __init__(self,
                 split_file,
                 split,
                 root,
                 num_classes,
                 task='class',
                 frames=80,
                 gamma_tau=5,
                 crops=1,
                 use_contrastive_loss= False,
                 spatial_transform=None,
                 data_augmentation=None):

        self.data = make_dataset(split_file, split, root, num_classes)
        self.split_file = split_file
        self.root = root
        self.frames = frames * 2
        self.gamma_tau = gamma_tau * 2
        self.loader = get_default_video_loader()
        self.spatial_transform = spatial_transform
        self.crops = crops
        self.split = split
        self.task = task
        self.data_augmentation = data_augmentation
        self.use_contrastive_loss = use_contrastive_loss

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        if self.crops > 1:
            self.split = 'validation'

        vid, label, nf = self.data[index]
        if self.split == 'validation':
            frames = nf
            start_f = 1
        else:
            frames = self.frames
            start_f = random.randint(1,nf-(self.frames+1))

        stride_f = self.gamma_tau
        imgs = load_rgb_frames(self.root, vid, start_f, frames, stride_f, self.loader)
        label = label[:, start_f-1:start_f-1+frames:1] #stride_f
        label = torch.from_numpy(label)

        if self.task == 'class':
            label = torch.max(label, dim=1)[0] # C T --> C

        assert (self.spatial_transform != None)

        # When training with contrastive loss
        if self.use_contrastive_loss:
            img_list_a = [self.spatial_transform(Image.fromarray(one_img)) for one_img in imgs]
            img_list_b = [self.spatial_transform(Image.fromarray(one_img)) for one_img in imgs]

            img_list_a = torch.stack(img_list_a, 0).permute(1, 0, 2, 3)  # T C H W --> C T H W
            img_list_b = torch.stack(img_list_b, 0).permute(1, 0, 2, 3)

            if self.split == 'validation' and self.task == 'class':  # self.crops > 1:
                clips_a = [img_list_a[:, :self.frames // self.gamma_tau, ...]]
                clips_b = [img_list_b[:, :self.frames // self.gamma_tau, ...]]

                clips_a = torch.stack(clips_a, 0)
                clips_b = torch.stack(clips_b, 0)

            else:
                clips_a = img_list_a
                clips_b = img_list_b

            return clips_a, clips_b, label

        # When training without contrastive loss
        else:
            imgs_l = [self.spatial_transform(Image.fromarray(one_img)) for one_img in imgs]
            imgs_l = torch.stack(imgs_l, 0).permute(1, 0, 2, 3)

            if self.split == 'validation' and self.task == 'class':  # self.crops > 1:
                step = int((imgs_l.shape[1] - 1 - self.frames // self.gamma_tau) // (self.crops - 1))

                if step == 0:
                    clips = [imgs_l[:, :self.frames // self.gamma_tau, ...] for i in range(self.crops)]
                    clips = torch.stack(clips, 0)
                else:
                    clips = [imgs_l[:, i:i + self.frames // self.gamma_tau, ...] for i in range(0, step * self.crops, step)]
                    clips = torch.stack(clips, 0)
            else:
                clips = imgs_l

            return clips, label


    def __len__(self):
        return len(self.data)


def custom_collate_fn(batch):
    "Pads data and puts it into a tensor of same dimensions"
    max_len_clips = 0
    max_len_labels = 0
    for b in batch:
        if b[0].shape[1] > max_len_clips:
            max_len_clips = b[0].shape[1]
        if b[1].shape[1] > max_len_labels:
            max_len_labels = b[1].shape[1]

    new_batch = []
    for b in batch:
        clips = np.zeros((b[0].shape[0], max_len_clips, b[0].shape[2], b[0].shape[3]), np.float32)
        label = np.zeros((b[1].shape[0], max_len_labels), np.float32)
        mask = np.zeros((max_len_labels), np.float32)

        clips[:,:b[0].shape[1],:,:] = b[0]
        label[:,:b[1].shape[1]] = b[1]
        mask[:b[1].shape[1]] = 1

        new_batch.append([torch.from_numpy(clips), torch.from_numpy(label), torch.from_numpy(mask)])

    return default_collate(new_batch)
