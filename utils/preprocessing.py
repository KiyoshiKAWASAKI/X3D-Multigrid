# Original author: Dawei Du
# Modified by Jin Huang
# Note on 06/10: only consider training process for now



import json
import os
import pandas as pd
import cv2
from tqdm import trange
import pdb
import os
from os import listdir
from os.path import isfile, join
import pyunpack
import sys



def txt2csv(src, dst):
    """

    """
    df = pd.read_csv(src, delimiter=' ')
    df.to_csv(dst, index=False, header=False)




def set_default(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError



class SetEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        return json.JSONEncoder.default(self, obj)



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




def process_ucf_txt2csv(src_txt_path, save_csv_path):
    """
    06/09
    This is the function used to process partition files from
    /data/dawei.du/datasets/NoveltyActionRecoSplits/UCF101_NoveltySplits/

    Note: to match the original format, labels start from 1
    """
    # Load txt file
    data = pd.read_csv(src_txt_path, names=["video"])

    # Add a new column with label part only
    data['label'] = data.video.str[:-12]
    unique_keys = data.label.unique()

    # Generate a dictionary for these keys
    key_dict ={}
    for i in range(len(unique_keys)):
        key_dict[unique_keys[i]] = i+1

    # Check all entries and assign labels
    data["num_label"] = data["label"].map(key_dict)

    # Remove index and word label columns
    data_final = data[["video", "num_label"]]

    # Save dataframe into csv without header and index
    data_final.to_csv(save_csv_path, header=False, index=False)



def process_hmdb_txt2csv(src_txt_path, save_csv_path):
    """
    06/10
    This is the function used to process partition files from
    /data/dawei.du/datasets/NoveltyActionRecoSplits/HMDB51_NoveltySplits/

    Note: to match the original format, labels start from 1
    """
    # Load txt file
    data = pd.read_csv(src_txt_path, names=["video"])

    # Add a new column with label part only
    # TODO: split the names with "/"
    f = lambda x: len(x['video'].split("/")) - 1
    # reviews["disappointed"] = reviews.apply(f, axis=1)
    data['label'] = data.apply(f, axis=1)
    # data['label'] = data.video.split("/")[0]
    unique_keys = data.label.unique()

    # Generate a dictionary for these keys
    key_dict ={}
    for i in range(len(unique_keys)):
        key_dict[unique_keys[i]] = i+1

    # Check all entries and assign labels
    data["num_label"] = data["label"].map(key_dict)

    # Remove index and word label columns
    data_final = data[["video", "num_label"]]

    # Save dataframe into csv without header and index
    data_final.to_csv(save_csv_path, header=False, index=False)




def convert_ucf_to_dict(dataset_path, csv_path, gen_test):
    """
    Generate the keys and labels for training and validation
    """

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

    # print(unique_list) # Correct

    # for i in range(len(unique_list)):
    #     base_name, class_name = unique_list[i].split("@")
    #     video_path = os.path.join(dataset_path, base_name)
    #
    #     print(video_path)

    if gen_test:
        for i in range(len(unique_list)):
            try:
                base_name, class_name = unique_list[i].split("@")
                video_path = os.path.join(dataset_path, base_name)

                # TODO: process test data into 2 parts
                # Train: for feedback
                # Validation: for test
                if i % 2 != 0:  # 50% feedback data
                    train_keys.append(video_path)
                    train_key_labels.append(class_name)
                else:  # 50%
                    val_keys.append(video_path)
                    val_key_labels.append(class_name)
            except:
                print(i, unique_list[i].split("@"))

        # return train_keys, train_key_labels

    else:
        for i in range(len(unique_list)):
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



def get_hmdb_labels(label_csv_path):
    data_original = pd.read_csv(label_csv_path, delimiter=',', names=["video", "num_label"])
    data = data_original[["video"]]

    data[['label', 'path']] = data['video'].str.split('/', expand=True)
    unique_keys = data.label.unique()

    # print(unique_keys)

    return unique_keys.tolist()




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




def convert_TA2_to_json(train_csv_path,
                        train_mode,
                        video_dir_path,
                        dst_json_path,
                        gen_test,
                        process_hmdb51=False):
    if gen_test:
        dst_data_feedback = {}
        dst_data_test = {}

        dst_data_feedback['database'] = {}
        dst_data_test['database'] = {}

        train_database, val_database = {}, {}
    else:
        dst_data = {}
        dst_data['database'] = {}
        train_database, val_database = {}, {}

    for k in range(len(train_csv_path)):
        if gen_test:
            if train_mode[0] == 'ucf101':
                labels, label_map = load_ucf_labels(train_csv_path[k])


                train_keys, val_keys, train_key_labels, val_key_labels = convert_ucf_to_dict(video_dir_path[0],
                                                                                             train_csv_path[k],
                                                                                             gen_test=True)

                dst_data_feedback['labels'] = labels
                dst_data_test['labels'] = labels

            elif train_mode[0] == "hmdb51":
                labels = get_hmdb_labels(train_csv_path[k])
                train_keys, val_keys, train_key_labels, val_key_labels= convert_ucf_to_dict(video_dir_path[0],
                                                                                             train_csv_path[k],
                                                                                             gen_test=True)
                dst_data_feedback['labels'] = labels
                dst_data_test['labels'] = labels

            else:
                pass
                # train_keys_, val_keys_, train_key_labels_, val_key_labels_ = convert_kinetics_to_dict(video_dir_path[k],
                #                                                                                       train_csv_path[k],
                #                                                                                       label_map)
                # train_keys = train_keys + train_keys_
                # val_keys = val_keys + val_keys_
                # train_key_labels = train_key_labels + train_key_labels_
                # val_key_labels = val_key_labels + val_key_labels_

        else:
            if train_mode[0] == 'ucf101':
                labels, label_map = load_ucf_labels(train_csv_path[k])
                train_keys, val_keys, train_key_labels, val_key_labels = convert_ucf_to_dict(video_dir_path[0],
                                                                                             train_csv_path[k],
                                                                                             gen_test=False)
                dst_data['labels'] = labels

            elif train_mode[0] == "hmdb51":
                labels =  get_hmdb_labels(train_csv_path[k])
                train_keys, val_keys, train_key_labels, val_key_labels = convert_ucf_to_dict(video_dir_path[0],
                                                                                             train_csv_path[k],
                                                                                             gen_test=False)
                dst_data['labels'] = labels

            else:
                train_keys_, val_keys_, train_key_labels_, val_key_labels_ = convert_kinetics_to_dict(video_dir_path[k],
                                                                                                      train_csv_path[k],
                                                                                                      label_map)
                train_keys = train_keys + train_keys_
                val_keys = val_keys + val_keys_
                train_key_labels = train_key_labels + train_key_labels_
                val_key_labels = val_key_labels + val_key_labels_

    # HMDB51 has subfolders for each class, so we need to process in a different way
    if process_hmdb51:
        if gen_test:
            for i in range(len(train_keys)):
                old_key = train_keys[i]
                key = video_dir_path[0] + train_key_labels[i] + "/" + old_key.split("/")[-1]
                train_database[key] = {}
                train_database[key]['subset'] = 'feedback_set'
                train_database[key]['annotations'] = {'label': train_key_labels[i]}

            for i in range(len(val_keys)):
                old_key = val_keys[i]
                key = video_dir_path[0] + val_key_labels[i] + "/" + old_key.split("/")[-1]
                val_database[key] = {}
                val_database[key]['subset'] = 'test_set'
                val_database[key]['annotations'] = {'label': val_key_labels[i]}

            dst_data_feedback['database'].update(train_database)
            dst_data_test['database'].update(val_database)


        else:
            for i in range(len(train_keys)):
                old_key = train_keys[i]
                key = video_dir_path[0] + train_key_labels[i] + "/" + old_key.split("/")[-1]
                train_database[key] = {}
                train_database[key]['subset'] = 'training'
                train_database[key]['annotations'] = {'label': train_key_labels[i]}

            for i in range(len(val_keys)):
                old_key = val_keys[i]
                key = video_dir_path[0] + val_key_labels[i] + "/" + old_key.split("/")[-1]
                val_database[key] = {}
                val_database[key]['subset'] = 'validation'
                val_database[key]['annotations'] = {'label': val_key_labels[i]}

            dst_data['database'].update(train_database)
            dst_data['database'].update(val_database)



    else:
        if gen_test:
            for i in range(len(train_keys)):
                key = train_keys[i]
                train_database[key] = {}
                train_database[key]['subset'] = 'feedback_set'
                label = train_key_labels[i]
                train_database[key]['annotations'] = {'label': label}

            for i in range(len(val_keys)):
                key = val_keys[i]
                val_database[key] = {}
                val_database[key]['subset'] = 'test_set'
                label = val_key_labels[i]
                val_database[key]['annotations'] = {'label': label}

            dst_data_feedback['database'].update(train_database)
            dst_data_test['database'].update(val_database)
        else:
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



    if gen_test:
        cnt = 0
        print("Processing feedback set")
        for video_path, frame_range in dst_data_feedback['database'].items():
            cnt += 1
            if cnt % 100 == 0:
                print("parsing sample {}/{}...".format(cnt, len(train_keys)))
            try:
                n_frames = get_n_frames(video_path)
                frame_range['annotations']['segment'] = (1, n_frames)
            except:
                continue

        cnt = 0
        print("Processing test set")
        for video_path, frame_range in dst_data_test['database'].items():
            cnt += 1
            if cnt % 100 == 0:
                print("parsing sample {}/{}...".format(cnt, len(val_keys)))
            try:
                n_frames = get_n_frames(video_path)
                frame_range['annotations']['segment'] = (1, n_frames)
            except:
                continue

        with open(dst_json_path[0], 'w') as dst_file_feedback:
            json.dump(dst_data_feedback, dst_file_feedback, default=set_default)
            print("data saved to %s" % dst_json_path[0])

        with open(dst_json_path[1], 'w') as dst_file_test:
            json.dump(dst_data_test, dst_file_test, default=set_default)
            print("data saved to %s" % dst_json_path[1])

    else:
        cnt = 0

        for video_path, frame_range in dst_data['database'].items():
            cnt += 1
            if cnt % 100 == 0:
                print("parsing sample {}/{}...".format(cnt, len(train_keys)))
            try:
                n_frames = get_n_frames(video_path)
                frame_range['annotations']['segment'] = (1, n_frames)
            except:
                continue

        with open(dst_json_path, 'w') as dst_file:
            # json.dump(dst_data, dst_file, cls=SetEncoder)
            json.dump(dst_data, dst_file, default=set_default)
            # json.dump(dst_data, dst_file)
            print("data saved to %s" % dst_json_path)




def get_hmdb51_data(data_dir):
    """
    Unrar each rar file into their own folder for HMDB51 dataset
    # TODO: the unpacked folders have 2 layers
    """
    # List all the files in data directory
    rar_file_list = [f for f in listdir(data_dir) if isfile(join(data_dir, f))]

    for one_rar in rar_file_list:
        print("Processing: %s" % one_rar)
        # Make a folder for it
        class_name = one_rar.split(".")[0]
        os.mkdir(data_dir + "/" + class_name)

        # Unpack the RAR file
        print("Unpacking RAR file")
        rar_path = data_dir + "/" + one_rar
        pyunpack.Archive(rar_path).extractall(data_dir + "/" + class_name)




if __name__ == '__main__':
    dataset_name = "hmdb"
    # dataset_name = "ucf101"

    if dataset_name == "ucf101":
        train_mode = ['ucf101']
        video_path = ['/afs/crc.nd.edu/user/j/jhuang24/scratch_51/kitware_internship/UCF101/']

        dir_path = '/afs/crc.nd.edu/user/j/jhuang24/scratch_51/kitware_internship/ucf101_npy_json/ta2_10_folds/0_crc/' #Path of label directory
        dst_path = '/afs/crc.nd.edu/user/j/jhuang24/scratch_51/kitware_internship/ucf101_npy_json/ta2_10_folds/0_crc/' # Directory path of dst json file.
        dst_json_path = dst_path + 'ta2_10_folds_partition_0.json'

        # train_known_txt_path = "/data/dawei.du/datasets/NoveltyActionRecoSplits/UCF101_NoveltySplits/0/seen_training_filelist_0.txt"
        # test_known_txt_path = "/data/dawei.du/datasets/NoveltyActionRecoSplits/UCF101_NoveltySplits/0/seen_test_filelist_0.txt"
        # test_unknown_txt_path = "/data/dawei.du/datasets/NoveltyActionRecoSplits/UCF101_NoveltySplits/0/unseen_filelist_0.txt"

        train_known_csv_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/kitware_internship/" \
                               "ucf101_npy_json/ta2_10_folds/0/train_known_0.csv"
        test_known_csv_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/kitware_internship/" \
                              "ucf101_npy_json/ta2_10_folds/0/test_known_0.csv"
        test_unknown_csv_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/kitware_internship/" \
                                "ucf101_npy_json/ta2_10_folds/0/test_unknown_0.csv"

        # process_ucf_txt2csv(src_txt_path=train_known_txt_path,
        #                     save_csv_path=train_known_csv_path)
        # process_ucf_txt2csv(src_txt_path=test_known_txt_path,
        #                     save_csv_path=test_known_csv_path)
        # process_ucf_txt2csv(src_txt_path=test_unknown_txt_path,
        #                     save_csv_path=test_unknown_csv_path)

        # Generate training Json file
        train_csv_path = [train_known_csv_path]
        test_csv_path = []

        convert_TA2_to_json(train_csv_path=train_csv_path,
                            train_mode=train_mode,
                            video_dir_path=video_path,
                            dst_json_path=dst_json_path,
                            gen_test=False)

        test_known_csv = [test_known_csv_path]
        test_unknown_csv = [test_unknown_csv_path]

        dst_test_known_json_path_feedback = dst_path + 'ta2_partition_0_test_known_feedback.json'
        dst_test_known_json_path_test = dst_path + 'ta2_partition_0_test_known_test.json'

        dst_test_unknown_json_path_feedback = dst_path + 'ta2_partition_0_test_unknown_feedback.json'
        dst_test_unknown_json_path_test = dst_path + 'ta2_partition_0_test_unknown_test.json'

        # convert_TA2_to_json(train_csv_path=test_known_csv,
        #                     train_mode=train_mode,
        #                     video_dir_path=video_path,
        #                     dst_json_path=[dst_test_known_json_path_feedback,
        #                                    dst_test_known_json_path_test],
        #                     gen_test=True,
        #                     process_hmdb51=False)
        #
        # convert_TA2_to_json(train_csv_path=test_unknown_csv,
        #                     train_mode=train_mode,
        #                     video_dir_path=video_path,
        #                     dst_json_path=[dst_test_unknown_json_path_feedback,
        #                                    dst_test_unknown_json_path_test],
        #                     gen_test=True,
        #                     process_hmdb51=False)

    elif dataset_name == "hmdb":
        hmdb51_rar = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/kitware_internship/hmdb51/HMDB51"
        # get_hmdb51_data(data_dir=hmdb51_rar) # Uncomment this when doawloading the

        train_mode = ['hmdb51']
        video_path = ['/afs/crc.nd.edu/user/j/jhuang24/scratch_51/kitware_internship/hmdb51/HMDB51/']

        dir_path = '/afs/crc.nd.edu/user/j/jhuang24/scratch_51/kitware_internship/hmdb51/npy_json/0_crc/'  # Path of label directory
        dst_path = '/afs/crc.nd.edu/user/j/jhuang24/scratch_51/kitware_internship/hmdb51/npy_json/0_crc/'  # Directory path of dst json file.
        dst_json_path = dst_path + 'ta2_10_folds_partition_0.json'

        # train_known_txt_path = "/data/dawei.du/datasets/NoveltyActionRecoSplits/HMDB51_NoveltySplits/0/seen_training_filelist_0.txt"
        # test_known_txt_path = "/data/dawei.du/datasets/NoveltyActionRecoSplits/HMDB51_NoveltySplits/0/seen_test_filelist_0.txt"
        # test_unknown_txt_path = "/data/dawei.du/datasets/NoveltyActionRecoSplits/HMDB51_NoveltySplits/0/unseen_filelist_0.txt"

        train_known_csv_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/kitware_internship/hmdb51/npy_json/0/train_known_0.csv"
        test_known_csv_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/kitware_internship/hmdb51/npy_json/0/test_known_0.csv"
        test_unknown_csv_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/kitware_internship/hmdb51/npy_json/0/test_unknown_0.csv"

        # Uncomment this when generating CSV files for HMDB51
        # process_hmdb_txt2csv(src_txt_path=train_known_txt_path,
        #                      save_csv_path=train_known_csv_path)
        # process_hmdb_txt2csv(src_txt_path=test_known_txt_path,
        #                      save_csv_path=test_known_csv_path)
        # process_hmdb_txt2csv(src_txt_path=test_unknown_txt_path,
        #                      save_csv_path=test_unknown_csv_path)

        train_csv_path = [train_known_csv_path]

        # convert_TA2_to_json(train_csv_path=train_csv_path,
        #                     train_mode=train_mode,
        #                     video_dir_path=video_path,
        #                     dst_json_path=dst_json_path,
        #                     gen_test=False,
        #                     process_hmdb51=True)

        test_known_csv = [test_known_csv_path]
        test_unknown_csv = [test_unknown_csv_path]

        # dst_test_known_json_path = dst_path + 'ta2_partition_0_test_known.json'
        # dst_test_unknown_json_path = dst_path + 'ta2_partition_0_test_unknown.json'

        dst_test_known_json_path_feedback = dst_path + 'ta2_partition_0_test_known_feedback.json'
        dst_test_known_json_path_test = dst_path + 'ta2_partition_0_test_known_test.json'

        dst_test_unknown_json_path_feedback = dst_path + 'ta2_partition_0_test_unknown_feedback.json'
        dst_test_unknown_json_path_test = dst_path + 'ta2_partition_0_test_unknown_test.json'

        convert_TA2_to_json(train_csv_path=test_known_csv,
                            train_mode=train_mode,
                            video_dir_path=video_path,
                            dst_json_path=[dst_test_known_json_path_feedback,
                                           dst_test_known_json_path_test],
                            gen_test=True,
                            process_hmdb51=True)

        convert_TA2_to_json(train_csv_path=test_unknown_csv,
                            train_mode=train_mode,
                            video_dir_path=video_path,
                            dst_json_path=[dst_test_unknown_json_path_feedback,
                                           dst_test_unknown_json_path_test],
                            gen_test=True,
                            process_hmdb51=True)


