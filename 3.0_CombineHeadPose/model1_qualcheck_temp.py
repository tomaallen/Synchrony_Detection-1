# %%
import os
from glob import glob
import csv
import subprocess, shlex
from filter_dict import filter_dict
import json
import pandas as pd
import numpy as np
from reaching_const import KEYPOINTS_DICT
from tqdm import tqdm

from pathlib import Path
import sys
sys.path.append(str(Path(os.getcwd()).parent))
import settings
import data_quality_check as dqc

import multiprocessing as mp
import time
 
# TODO: add multicore processing

def analysis_sequence(filepath: str):
    tp, ppt, vid = filepath.split("\\")[-3:]

    f = open(filepath)
    dict_initial = json.load(f) # initial dictionary
    f.close()

    ai_dyad, confident = filter_dict(dict_initial, conf_threshold=0.3) # FIXME: rename function
    # ai_dyad = whether mum and baby are in the frame
    # confident = which frames keypoint is above confidence threshold for mum and baby
    frame_check = pd.DataFrame([[x] + list(y) for x, y in zip(ai_dyad, confident)],
                    columns = ['DyadPresent'] + list(KEYPOINTS_DICT.values()))
    frame_check.to_csv(settings.FRAME_CHECKS / (tp + "_" + ppt + "_" + vid + ".csv"))

    return [[filepath, ppt.strip('PID'), tp.strip('timepoint'), vid, len(dict_initial), np.mean(ai_dyad)]]


if __name__ == '__main__':

    start = time.time()
    os.makedirs(settings.ANALYSIS_FOLDER, exists_ok=True)
    os.makedirs(settings.FRAME_CHECK, exists_ok=True)

    # create data quality check file
    DATA_QUALITY = os.path.join(settings.ANALYSIS_FOLDER / "data_quality_summary.csv")
    with open(DATA_QUALITY, 'w', newline="") as f:
        write = csv.writer(f)
        write.writerows([['Filepath', 'Participant', 'Timepoint', 'Total frames', 'Good frames']])

    # create frame check folder
    if not os.path.isdir(settings.FRAME_CHECKS):
        os.mkdir(settings.FRAME_CHECKS)

    # get info on analysis, copy face csv to pose folder and create combined json
    filepaths = glob(str(settings.FOLDER / "*\\*\\*.json"))

    print('Analysing {} files'.format(len(filepaths)))
    pool = mp.Pool(mp.cpu_count())
    # results = pool.map(analysis_sequence, filepaths) # TODO: use imap and tqdm for a progress bar
    results = list(tqdm(pool.imap(analysis_sequence, filepaths), total=len(filepaths)))

    for i, (filepath, result) in enumerate(zip(filepaths, results)):
        tp, ppt, vid = filepath.split('\\')[-3:]

        # write to data quality file for camera selection
        with open(DATA_QUALITY, 'a', newline="") as f:
            write = csv.writer(f)
            write.writerows(result)

    # choose the camera with the most good frames for each participant and timepoint
    best_cam_list = dqc.get_best_cams([['LShoulder', 'LElbow'],
                                        ['RShoulder', 'RElbow'],
                                        ['Nose', 'LEar'],
                                        ['Nose', 'REar']])
    best_cam_list.to_csv(settings.BEST_CAMERAS)

    end = time.time()
    print('Runtime: {}sec'.format(round(end - start)))

    

# %%
