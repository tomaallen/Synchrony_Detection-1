import shutil
import os
import argparse
import csv
import subprocess, shlex
from filter_dict import filter_dict
import json
import pandas as pd
import re
import numpy as np
from reaching_const import KEYPOINTS_DICT
import synapseclient

from pathlib import Path
import sys
sys.path.append(str(Path(os.getcwd()).parent))
import settings

import time

if __name__ == '__main__':
    
    start = time.time()

    # reset info file
    analysis_info = os.path.join(settings.ANALYSIS_FOLDER, "analysis_info.csv")
    if not os.path.isdir(settings.ANALYSIS_FOLDER): # make analysis info folder if does not exist
        os.mkdir(settings.ANALYSIS_FOLDER)
    with open(analysis_info, 'w', newline="") as f:
        write = csv.writer(f)
        write.writerows([['Filename','Face_attempted','Face_complete','Pose_attempted',
                          'Pose_complete','Csv_copied','Json_combined',
                          'Plots_run','Parameters_run']])

    # create data quality check file
    data_quality = os.path.join(settings.ANALYSIS_FOLDER, "data_quality_summary.csv")
    with open(data_quality, 'w', newline="") as f:
        write = csv.writer(f)
        write.writerows([['Filename', 'Filepath', 'Total frames', 'Good frames']])

    # create frame check folder
    if not os.path.isdir(settings.FRAME_CHECKS):
        os.mkdir(settings.FRAME_CHECKS)
    
    # get info on analysis, copy face csv to pose folder and create combined json
    for file in os.listdir(settings.FOLDER):
        
        # initialise info variables
        face_attempted = face_completed = face_runs_check = pose_attempted \
            = pose_completed = csv_copied = json_combined = False
        
        if file.endswith('.mp4'): # find all videos in settings.FOLDER
        
            # variables for copying face csv to pose folder
            vid = os.path.splitext(file)[0] # remove mp4 extension
            face_csv_file = os.path.join(settings.HEAD_FOLDER, vid, vid + '.csv')
            pose_facecsv_folder = os.path.join(settings.POSE_FOLDER, vid, 'csv_face')
            target_file = os.path.join(pose_facecsv_folder, vid + '.csv')
            
            # path for combined json file
            combined_json_path = os.path.join(settings.POSE_FOLDER, vid, 'json_files', vid + '-PD-combined_output.json')

            # check if face and pose detection have run
            face_attempted = os.path.isdir(os.path.join(settings.HEAD_FOLDER, vid))
            face_completed = face_attempted and os.path.exists(face_csv_file) # face completed if csv created
            pose_attempted = os.path.isdir(os.path.join(settings.POSE_FOLDER, vid))
            pose_completed = pose_attempted and \
                os.path.exists(os.path.join(settings.POSE_FOLDER, vid, 'json_files', 
                                            vid + '-PD-output.json')) # pose completed if json created
            
            
            # run combined analysis if face and pose completed
            if face_completed and pose_completed:
                
                # copy face csv to pose folder
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
                        + os.path.join(settings.POSE_FOLDER, vid)
                    print(shlex.split(command, posix = 0))
                    subprocess.check_call(shlex.split(command, posix = 0))
                
                json_combined = os.path.exists(combined_json_path)

                # data quality check
                print('######### Data quality check on: ' + os.path.basename(combined_json_path) + ' ######################\n')
                print(combined_json_path)

                f = open(combined_json_path)
                dict_initial = json.load(f) # initial dictionary
                f.close()
                
                ai_dyad, confident = filter_dict(dict_initial, conf_threshold=0.3) # FIXME: rename function
                # ai_dyad = whether mum and baby are in the frame
                # confident = which frames keypoint is above confidence threshold for mum and baby
                frame_check = pd.DataFrame([[x] + list(y) for x, y in zip(ai_dyad, confident)],
                                columns = ['DyadPresent'] + list(KEYPOINTS_DICT.values()))
                frame_check.to_csv(settings.FRAME_CHECKS / (vid + ".csv"))
                
                # write to data quality file for camera selection
                with open(data_quality, 'a', newline="") as f:
                    write = csv.writer(f)
                    write.writerows([[vid,
                                    combined_json_path,
                                    len(dict_initial),
                                    np.mean(ai_dyad)]])                          

            # write analysis info to csv file in settings.FOLDER
            with open(analysis_info, 'a', newline="") as f:
                write = csv.writer(f)
                write.writerows([[vid, face_attempted, face_completed, pose_attempted, 
                                  pose_completed, csv_copied, json_combined]])

    end = time.time()
    print('Runtime: {}sec'.format(round(end - start)))


# %%
