# Original author: Dawei Du
#
import json
import os
import pandas as pd
import cv2
from tqdm import trange
import pdb


def get_n_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_length = 0
    while(True):
        ret, frame = cap.read()
        if not ret:
            break
        frame_length += 1
    cap.release()

    return frame_length




def convert_ucf_to_dict(dataset_path, csv_path):
    data = pd.read_csv(csv_path, delimiter=',', header=None)
    train_keys, val_keys = [], []
    train_key_labels, val_key_labels = [], []
    tmp = []
    for i in range(data.shape[0]):
        slash_rows = data.iloc[i, 0].split('/')
        class_name = slash_rows[0]
        basename = slash_rows[1]
        tmp.append(basename+"@"+class_name)
    unique_list = sorted(set(tmp),key=tmp.index)
    
    for i in trange(len(unique_list)):
        base_name, class_name = unique_list[i].split("@")
        video_path = os.path.join(dataset_path, base_name)
        if i%10 != 0: # 90% training
            train_keys.append(video_path)
            train_key_labels.append(class_name)
        else: # 10% validation
            val_keys.append(video_path)
            val_key_labels.append(class_name)        

    return train_keys, val_keys, train_key_labels, val_key_labels


def load_ucf_labels(label_csv_path):
    data = pd.read_csv(label_csv_path, delimiter=',', header=None)
    labels = []
    exist_label = []
    label_map = {}
    for i in range(data.shape[0]):
        if data.iloc[i,1] not in exist_label:
            labels.append(data.iloc[i, 0].split('/')[0])
            exist_label.append(data.iloc[i,1])
            label_map[data.iloc[i, 1]] = data.iloc[i, 0].split('/')[0]
    return labels, label_map




def load_kinetics_labels(label_csv_path, label_map):
    data = pd.read_csv(label_csv_path, delimiter=',', header=None)
    labels = []
    exist_label = []
    for i in range(data.shape[0]):
        if data.iloc[i,1] not in exist_label:
            labels.append(data.iloc[i, 0].split('/')[0])
            exist_label.append(data.iloc[i,1])
            label_map[data.iloc[i, 0]] = data.iloc[i, 0].split('/')[0]
    return labels, label_map




def convert_kinetics_to_dict(dataset_path, csv_path, label_map):
    data = pd.read_csv(csv_path, delimiter=',', header=None)
    train_keys, val_keys = [], []
    train_key_labels, val_key_labels = [], []
    tmp = []
    for i in range(data.shape[0]):
        slash_rows = data.iloc[i, 0].split('/')
        class_name = label_map[data.iloc[i,1]]
        basename = 'X'+slash_rows[0]+'.mp4'
        tmp.append(basename+"@"+class_name)
    unique_list = sorted(set(tmp),key=tmp.index)

    for i in trange(len(unique_list)):
        base_name, class_name = unique_list[i].split("@")
        video_path = os.path.join(dataset_path, base_name)
        if os.path.exists(video_path):
            if i%10 != 0: # 90% training
                train_keys.append(video_path)
                train_key_labels.append(class_name)
            else: # 10% validation
                val_keys.append(video_path)
                val_key_labels.append(class_name)       

    return train_keys, val_keys, train_key_labels, val_key_labels




def convert_TA2_to_json(train_csv_path, train_mode, video_dir_path, dst_json_path):
    dst_data = {}
    dst_data['database'] = {}
    train_database, val_database = {}, {}
    for k in range(len(train_csv_path)):
        if train_mode[k] == 'ucf101':
            labels, label_map = load_ucf_labels(train_csv_path[k])
            train_keys, val_keys, train_key_labels, val_key_labels = convert_ucf_to_dict(video_dir_path[k], train_csv_path[k])
            dst_data['labels'] = labels
        else:
            train_keys_, val_keys_, train_key_labels_, val_key_labels_ = convert_kinetics_to_dict(video_dir_path[k], train_csv_path[k], label_map)
            train_keys = train_keys + train_keys_
            val_keys = val_keys + val_keys_
            train_key_labels = train_key_labels + train_key_labels_
            val_key_labels = val_key_labels + val_key_labels_

    # save database
    for i in range(len(train_keys)):
        key = train_keys[i]
        train_database[key] = {}
        train_database[key]['subset'] = 'training'
        label = train_key_labels[i]
        train_database[key]['annotations'] = {'label': label}

    for i in range(len(val_keys)):
        key = val_keys[i]
        val_database[key] = {}
        val_database[key]['subset'] = 'validation'
        label = val_key_labels[i]
        val_database[key]['annotations'] = {'label': label}

    dst_data['database'].update(train_database)
    dst_data['database'].update(val_database)
    cnt = 0
    for video_path, frame_range in dst_data['database'].items():
        cnt += 1
        if cnt % 1000 == 0:
            print("parsing sample {}/{}...".format(cnt, len(train_keys)))
        n_frames = get_n_frames(video_path)
        frame_range['annotations']['segment'] = (1, n_frames)

    with open(dst_json_path, 'w') as dst_file:
        json.dump(dst_data, dst_file)




if __name__ == '__main__':
    dir_path = '/data/dawei.du/datasets/TA2_splits/' # Path of label directory
    dst_path = '/data/jin.huang/ucf101_npy_json/' # Directory path of dst json file.
    dst_json_path = dst_path + 'ucf101.json'
    # train_mode = ['ucf101', 'kinetics']
    train_mode = ['ucf101']

    video_path = ['/data/dawei.du/datasets/UCF101/'] # Path of video directory
    # video_path.append('/data/dawei.du/datasets/kinetics/')

    train_csv_path = [dir_path + 'ucf101_train_knowns_revised.csv']
    # train_csv_path.append(dir_path + 'kinetics_train_knowns_revised.csv')

    convert_TA2_to_json(train_csv_path, train_mode, video_path, dst_json_path)

