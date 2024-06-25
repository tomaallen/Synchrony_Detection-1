import cv2
import mmcv
import subprocess
import os

import pandas as pd
from tqdm import tqdm
import numpy as np

from distutils.dir_util import copy_tree
import shutil
from pathlib import Path
import argparse

import sys
parent_dir = str(Path(os.getcwd()).parent)
sys.path.append(parent_dir)
import settings


def get_all_files_recursively_by_ext(root, ext):
    found = []
    for path in Path(root).rglob('*.{}'.format(ext)):
        found.append(str(path))
    return sorted(found)


def ensure_dir(file_path):
    directory = file_path
    if file_path[-3] == "." or file_path[-4] == ".":
        directory = os.path.dirname(file_path)
    Path(directory).mkdir(parents=True, exist_ok=True)


def run_yolo_on_videos(input_path, output_path, yolo_exp_name, yolo_py_path, yolo_weight_path):
    ensure_dir(output_path)

    yolo_label_directory = os.path.join("runs", "detect")

    yolo_command = "python {py} --weights {yolo_weight} --conf 0.25 --img-size 640 --source {input} --save-txt --save-conf --name {yolo_exp_name}".format(
        py=yolo_py_path, yolo_weight=yolo_weight_path, input=input_path, yolo_exp_name=yolo_exp_name)

    subprocess.call(yolo_command, shell=True)

    copy_tree(yolo_label_directory, output_path)
    shutil.rmtree(yolo_label_directory)


def gather_results(txt_directory, output_path, length):
    label_txts = get_all_files_recursively_by_ext(txt_directory, "txt")
    label_txts = sorted(label_txts, key=lambda x: int(x.split(os.sep)[-1].split("_")[-1][:-4]))

    labeled_frame = sorted([int(txt.split(os.sep)[-1].split("_")[-1][:-4]) - 1 for txt in label_txts])

    center_cords = np.ones((length, 4)) * -1
    for label_txt in tqdm(label_txts, total=len(label_txts)):
        label_df = pd.read_csv(label_txt, header=None, sep=" ")
        idx = int(label_txt.split(os.sep)[-1].split("_")[-1].split(".txt")[0]) - 1
        bboxes_mom = label_df.loc[(label_df[0] == 0) & (label_df[1] < 0.5)] # mom bboxes on left
        bboxes_baby_l = label_df.loc[(label_df[0] == 1) & (label_df[1] < 0.5)] # baby bboxes on left
        bboxes_baby_r = label_df.loc[(label_df[0] == 1) & (label_df[1] >= 0.5)] # baby bboxes on right

        cx_mom, cy_mom, cx_baby, cy_baby = -1, -1, -1, -1

        if len(bboxes_mom):
            # if at least 1 mum box present - select highest confidence
            _, x_float_mom, y_float_mom, _, _, _ = bboxes_mom.sort_values(5, ascending=False).iloc[0] # takes highest confidence
            cx_mom, cy_mom = x_float_mom, y_float_mom
        elif len(bboxes_baby_l):
            # if no mum bbox present choose baby box with highest confidence
            # on the left hand side of the screen as mum bbox
            _, x_float_mom, y_float_mom, _, _, _ = bboxes_baby_l.sort_values(5, ascending=False).iloc[0] # takes highest confidence
            cx_mom, cy_mom = x_float_mom, y_float_mom
        if len(bboxes_baby_r):
            # select baby box with highest confidence on the right hand side
            _, x_float_baby, y_float_baby, _, _, _ = bboxes_baby_r.sort_values(5, ascending=False).iloc[0]
            cx_baby, cy_baby = x_float_baby, y_float_baby

        center_cords[idx] = cx_mom, cy_mom, cx_baby, cy_baby

    data_df = pd.DataFrame(center_cords, columns=["cx_mom", "cy_mom", "cx_baby", "cy_baby"], index=None)
    data_df.to_csv(output_path, index=False)


def main(input_path, txt_output_directory, csv_output_path, yolo_exp_name, yolo_py_path, yolo_weight_path):
    if os.path.isdir(os.path.join(txt_output_directory, yolo_exp_name)):
        shutil.rmtree(os.path.join(txt_output_directory, yolo_exp_name))

    if os.path.isfile(csv_output_path):
        os.remove(csv_output_path)

    video = cv2.VideoCapture(input_path)
    length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    run_yolo_on_videos(input_path=input_path, output_path=txt_output_directory, yolo_exp_name=yolo_exp_name,
                       yolo_py_path=yolo_py_path, yolo_weight_path=yolo_weight_path)
    gather_results(txt_directory=txt_output_directory, output_path=csv_output_path, length=length)


if __name__ == "__main__":
    # edited
    parser = argparse.ArgumentParser(description='Parse args for head detection')
    parser.add_argument('--input', default=None,
                        type=str,
                        help='The path to the input video or folder.')
    parser.add_argument('--output', default=None, type=str,
                        help='Where to save the yolo raw output?')
    parser.add_argument('--yolo_exp_name', default='head_run', type=str,
                        help='Name the run, e.g., for LP006s video simply name it as "LP006", then a LP006.csv will be generate in txt_output_directory')
    parser.add_argument('--yolo_py_path', default='yolov7-main\\detect.py', type=str,
                        help='The path to detect.py from YoloV7 folder.')
    parser.add_argument('--yolo_weight_path', default='yolov7-main\\best.pt', type=str,
                        help='The path to the yolov7 weight you want to load.')

    opt = parser.parse_args()

    # to run on an individual file pass input argument
    if opt.input:
        if os.path.isfile(opt.input):
            csv_output_path = os.path.join(opt.output, opt.yolo_exp_name + ".csv")
            main(input_path=opt.input, txt_output_directory=opt.output, csv_output_path=csv_output_path,
                yolo_exp_name=opt.yolo_exp_name, yolo_py_path=opt.yolo_py_path, yolo_weight_path=opt.yolo_weight_path) #edited
          
        else:
            raise Exception("input does not exist")

    else: # to run on flat folder structure, do not pass input argument
        if os.path.isdir(settings.FOLDER):
            for file in os.listdir(settings.FOLDER):
                if file.endswith('.mp4'):
                    
                    # make save directory if it does not exist
                    if not os.path.isdir(settings.HEAD_FOLDER):
                        os.mkdir(settings.HEAD_FOLDER)
                        
                    # clear runs directory of previous unfinished analysis
                    if os.path.exists(os.path.join(os.getcwd(), 'runs')):
                        shutil.rmtree(os.path.join(os.getcwd(), 'runs'))
                        print(os.path.join(os.getcwd(), 'runs') + ' cleared')
                    
                    file_no_ext = os.path.splitext(file)[0]
                    vid_save_path = os.path.join(settings.HEAD_FOLDER, file_no_ext)
                    csv_save_path = os.path.join(vid_save_path, file_no_ext + ".csv")

                    if not os.path.isdir(vid_save_path): # if not run before
                        p_vid = os.path.join(settings.FOLDER, file)
                        main(input_path=p_vid,
                            txt_output_directory=vid_save_path,
                            csv_output_path=csv_save_path,
                            yolo_exp_name=file_no_ext,
                            yolo_py_path=opt.yolo_py_path,
                            yolo_weight_path=opt.yolo_weight_path)

        else:
            raise Exception('FOLDER in settings.py is not a directory')
