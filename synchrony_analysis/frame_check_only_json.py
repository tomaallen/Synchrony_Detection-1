import shutil
import os
import csv
import subprocess, shlex
from check_ppl_in_frames import check_ppl_in_frames
import json
import pandas as pd
import numpy as np
import constants
import multiprocessing as mp
from tqdm import tqdm
from glob import glob

from pathlib import Path
import sys
sys.path.append(str(Path(os.getcwd()).parent))
import settings

import time

def comb_analysis(combined_json_path, data_quality):
    
    # path for combined json file
    vid = os.path.splitext(os.path.basename(combined_json_path))[0] # video name extracted from file path

    # run frame check if not run yet
    frame_check_path = settings.FRAME_CHECKS / (vid + ".csv")

    # if not os.path.isfile(frame_check_path):

    # initial data quality check
    # print('######### Running initial data quality check on: ' + os.path.basename(combined_json_path) + ' ######################')
    f = open(combined_json_path)
    dict_initial = json.load(f) # initial dictionary
    f.close()
    
    # produce frame checks - whether each keypoint is present in each frame
    # one csv file per video in settings.FRAMECHECKS folder
    # ai_dyad = whether mum and baby are in the frame
    # confident = which frames keypoint is above confidence threshold for mum and baby
    ai_dyad, confident = check_ppl_in_frames(dict_initial, conf_threshold=0.3)
    frame_check = pd.DataFrame([[x] + list(y) for x, y in zip(ai_dyad, confident)],
                    columns = ['DyadPresent'] + list(constants.KEYPOINTS_DICT.values()))
    frame_check.to_csv(frame_check_path)

    return ['result'], [[vid, combined_json_path, len(dict_initial), np.mean(ai_dyad)]]




if __name__ == '__main__':
    
    start = time.time()

    # reset info file
    analysis_info = settings.ANALYSIS_FOLDER / "analysis_info.csv"
    os.makedirs(settings.ANALYSIS_FOLDER, exist_ok=True)
    with open(analysis_info, 'w', newline="") as f:
        write = csv.writer(f)
        write.writerows([['Filename','Face_attempted','Face_complete','Pose_attempted',
                          'Pose_complete','Csv_copied','Json_combined',
                          'Plots_run','Parameters_run']])

    # create data quality check summary file
    # Good Frames = proportion of frames where mother and infant are both detected
    data_quality = settings.ANALYSIS_FOLDER / "data_quality_summary.csv"
    with open(data_quality, 'w', newline='') as f:
        write = csv.writer(f)
        write.writerows([['Filename', 'Filepath', 'Total frames', 'Good frames']])

    # create frame check folder
    if not os.path.isdir(settings.FRAME_CHECKS):
        os.mkdir(settings.FRAME_CHECKS)
    
    # run multicore
    pool = mp.Pool(mp.cpu_count())
    params = [[file, data_quality] for file in glob(str(settings.FOLDER / "json_files" / "*.json"))]
    results = list(pool.starmap(comb_analysis, tqdm(params, total=len(params))))

    # write analysis info to csv file - which steps ran successfully for each video
    for _result, data_q in results:
        with open(analysis_info, 'a', newline="") as f:
            write = csv.writer(f)
            write.writerows([_result])
        
        # write data quality summary file
        with open(data_quality, 'a', newline="") as f:
            write = csv.writer(f)
            write.writerows(data_q) # proportion of frames where mother and infant are both detected 
        
    end = time.time()
    print('Runtime: {}sec'.format(round(end - start)))


# %%
