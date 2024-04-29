import os
from pathlib import Path
import re
import numpy as np
import pandas as pd
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import settings

import multiprocessing as mp
from tqdm import tqdm



def get_ppt(filename: str):
    return re.search(r'\d{8}', filename)[0] # XXX: edit this, BR: re.search(r'PID\d+', filename)[0]


def get_tp(filename: str): # requires manifest
    return re.search(r'\d+[Mm]', filename)[0] # XXX: edit this, BR: re.search(r'timepoint\d+', filename)[0]


def calc_quality_score(frame_check, checks):
    #     print('Data quality check for ' + frame_check)

    # for each csv file in settings.FRAME_CHECKS, generate a quality score from keypoint detections
    data_quality = pd.read_csv(os.path.join(settings.FRAME_CHECKS, frame_check), index_col=0)
    confident_frames = []
    for check in checks:
        _check_result = data_quality.loc[:, check].all(axis=1).values
        confident_frames.append(_check_result)
    # perfect_frames = np.array(confident_frames).all(axis=0) # perfect frames are those which meet all check criteria
    quality_score = np.mean(np.array(confident_frames))

    # try:
    ppt = get_ppt(frame_check) # get participant from filename
    tp = get_tp(frame_check) # get timepoint from filename
    # except:
    #     raise Exception('Please edit get_ppt() and get_tp() functions in data_quality_check.py to extract participant id and timepoint from filenames')

    return [os.path.splitext(frame_check)[0], ppt, tp, quality_score]


def get_best_cams(checks:list):
    # checks the quality of each video and returns a pd.Series of the best videos 
    # from each session
    # 
    # checks each csv in frames_checks folder
    # 
    # checks parameter is a nested list of groups of keypoints to check
    # e.g. [['LShoulder', 'LElbow'], ['RShoulder', 'RElbow']] checks
    # the number of frames where LShoulder and LElbow are both present
    # and the same for the right side. The mean of all checks is taken
    # to give a quality score. Passing a standard list will calculate the mean 
    # proportion of good frames across keypoints (no dependency on both keypoints 
    # being present).

    # run multicore
    pool = mp.Pool(mp.cpu_count())
    args = [(x, checks) for x in os.listdir(settings.FRAME_CHECKS)]
    results = list(pool.starmap(calc_quality_score, tqdm(args, total=len(args))))

    camera_scores = []
    for i, result in enumerate(results):
        camera_scores.append(result)

    # save camera quality scores for all cameras
    camera_scores = pd.DataFrame(camera_scores, columns=['Filename', 'ppt', 'tp', 'QualityScore'])

    # find the best camera for each session (participant and timepoint)
    best_cams_idx = list(camera_scores.groupby(['ppt', 'tp'])['QualityScore'].idxmax())
    best_cams = camera_scores.iloc[best_cams_idx]

    return camera_scores, best_cams


if __name__ == '__main__':
    pass

