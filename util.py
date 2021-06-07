import pandas as pd



ucf101_txt_dir = "/data/jin.huang/ucfTrainTestlist/"
ucf101_csv_save_dir = "/data/jin.huang/ucf101_npy_json/"
ucf101_file_list = ["trainlist01.txt", "trainlist02.txt", "trainlist03.txt"]

def txt2csv(src, dst):
    """

    """
    df = pd.read_csv(src, delimiter=' ')
    df.to_csv(dst, index=False, header=False)




if __name__ == '__main__':
    for one_file_name in ucf101_file_list:
        one_txt = ucf101_txt_dir + one_file_name
        file_base_name = one_file_name.split(".")[0]
        dst_csv_path = ucf101_csv_save_dir + file_base_name + ".csv"

        txt2csv(src=one_txt,
                dst=dst_csv_path)
