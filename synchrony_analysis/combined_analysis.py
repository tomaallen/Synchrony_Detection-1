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

from pathlib import Path
import sys
sys.path.append(str(Path(os.getcwd()).parent))
import settings

import time

def comb_analysis(file, data_quality):

    # initialise info variables
    face_attempted = face_completed = pose_attempted = pose_completed = csv_copied = json_combined = False
        
    # variables for copying face csv to pose folder
    vid = os.path.splitext(file)[0] # remove mp4 extension
    face_csv_file = settings.HEAD_FOLDER / vid / (vid + '.csv')
    pose_facecsv_folder = settings.POSE_FOLDER / vid / 'csv_face'
    target_file = pose_facecsv_folder / (vid + '.csv')
    
    # path for combined json file
    combined_json_path = settings.POSE_FOLDER / vid / 'json_files' / (vid + '-' + constants.JSON_COMBINED_FILE)

    # check if face and pose detection have run
    face_attempted = os.path.isdir(settings.HEAD_FOLDER / vid)
    face_completed = face_attempted and os.path.exists(face_csv_file) # face completed if csv created
    pose_attempted = os.path.isdir(settings.POSE_FOLDER / vid)
    pose_completed = pose_attempted and \
        os.path.exists(settings.POSE_FOLDER / vid / 'json_files' / (vid + '-' + constants.JSON_FILE)) # pose completed if json created
    
    # run combined analysis if face and pose completed
    if face_completed and pose_completed:
        
        # copy face csv to pose folder
        # NOTE: this step is not necessary but with the original design of model 1
        # to copy across into the pose folder is easier
        if not os.path.isdir(pose_facecsv_folder): 
            os.mkdir(pose_facecsv_folder)
            # print('pose facecsv folder was created: ' + pose_facecsv_folder)
        if not os.path.exists(target_file):
            shutil.copyfile(face_csv_file, target_file) # copy the face csv file
            # print(vid + ': csv copied  to ' + target_file)
            
        csv_copied = os.path.exists(target_file)
        
        # create combined json
        if not os.path.exists(combined_json_path):
            command = 'python create_combined_json.py --reach_dir ' \
                + str(settings.POSE_FOLDER / vid)
            print(shlex.split(command, posix = 0))
            subprocess.check_call(shlex.split(command, posix = 0))
        
        json_combined = os.path.exists(combined_json_path)

        # initial data quality check
        print('######### Running initial data quality check on: ' + os.path.basename(combined_json_path) + ' ######################')
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
        frame_check.to_csv(settings.FRAME_CHECKS / (vid + ".csv"))
        
        # write data quality summary file
        with open(data_quality, 'a', newline="") as f:
            write = csv.writer(f)
            write.writerows([[vid,
                            combined_json_path,
                            len(dict_initial), # total frames
                            np.mean(ai_dyad)]]) # proportion of frames where mother and infant are both detected                    

    return [vid, face_attempted, face_completed, pose_attempted, pose_completed, csv_copied, json_combined]




if __name__ == '__main__':
    
    start = time.time()

    # reset info file
    analysis_info = settings.ANALYSIS_FOLDER / "analysis_info.csv"
    if not os.path.isdir(settings.ANALYSIS_FOLDER): # make analysis info folder if does not exist
        os.mkdir(settings.ANALYSIS_FOLDER)
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
    params = [[file, data_quality] for file in os.listdir(settings.FOLDER) if file.endswith(".mp4")]
    results = list(pool.starmap(comb_analysis, tqdm(params, total=len(params))))

    # write analysis info to csv file - which steps ran successfully for each video
    for result in results:
        with open(analysis_info, 'a', newline="") as f:
            write = csv.writer(f)
            write.writerows([result])
        
    end = time.time()
    print('Runtime: {}sec'.format(round(end - start)))


# %%
