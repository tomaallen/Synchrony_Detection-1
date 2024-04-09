# %%
import os
import re
import numpy as np
import pandas as pd
import settings

def data_quality_check(df: pd.DataFrame, checks: list):
    confident_frames = []
    for check in checks:
        _check_result = df.loc[:, check].all(axis=1).values
        confident_frames.append(_check_result)
    perfect_frames = np.array(confident_frames).all(axis=0) # perfect frames are those which meet all check criteria
    quality_score = np.mean(np.array(confident_frames))

    return quality_score, perfect_frames # TODO: perfect frames not being used in the code currently

def get_ppt(filename: str):
   return re.search(r'PID\d+', filename)[0] # XXX: edit this, re.search(r'\d+', filename)[0]

def get_tp(filename:str):
    return re.search(r'timepoint\d+', filename)[0] # XXX: edit this, re.search(r'\d+a', filename)[0]

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

    camera_scores = []
    for frame_check in os.listdir(settings.FRAME_CHECKS):
        # for each csv file in settings.FRAME_CHECKS, generate a quality score from keypoint detections
        data_quality = pd.read_csv(os.path.join(settings.FRAME_CHECKS, frame_check), index_col=0)
        quality_score, _perfect_frames = data_quality_check(data_quality, checks)

        ppt = get_ppt(frame_check) # get participant from filename
        tp = get_tp(frame_check) # get timepoint from filename
        camera_scores.append([os.path.splitext(frame_check)[0], ppt, tp, quality_score])

    # save camera quality scores for all cameras
    camera_scores = pd.DataFrame(camera_scores, columns=['Filename', 'ppt', 'tp', 'QualityScore'])
    camera_scores.to_csv(settings.ANALYSIS_FOLDER / "model1_camera_scores.csv")
    # print(camera_scores)

    # find the best camera for each session (participant and timepoint)
    best_cams_idx = list(camera_scores.groupby(['ppt', 'tp'])['QualityScore'].idxmax())
    best_cams = camera_scores.iloc[best_cams_idx]
    best_cams.to_csv(str(settings.BEST_CAMERAS))

    return best_cams.Filename


# %%
