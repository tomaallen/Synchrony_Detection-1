import os
import re
import numpy as np
import pandas as pd
import settings

import multiprocessing as mp
from tqdm import tqdm

import synapseclient
import synapseutils

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


# # BR specific: get tp from synapse data

# # rename files with weird format and remove extensions
# def filekey_rename(x: str):
#     extensions = ['\.mp4', '\.mp4', '\.mov', '\.3gp']
#     for extension in extensions:
#         x = re.sub(extension, '', x)
#     x = x.lower().replace(' ','_').replace('ª','a')
#     return  x

# # synapse login and get list of PCI videos
# syn = synapseclient.login(authToken='eyJ0eXAiOiJKV1QiLCJraWQiOiJXN05OOldMSlQ6SjVSSzpMN1RMOlQ3TDc6M1ZYNjpKRU9VOjY0NFI6VTNJWDo1S1oyOjdaQ0s6RlBUSCIsImFsZyI6IlJTMjU2In0.eyJhY2Nlc3MiOnsic2NvcGUiOlsidmlldyIsImRvd25sb2FkIl0sIm9pZGNfY2xhaW1zIjp7fX0sInRva2VuX3R5cGUiOiJQRVJTT05BTF9BQ0NFU1NfVE9LRU4iLCJpc3MiOiJodHRwczovL3JlcG8tcHJvZC5wcm9kLnNhZ2ViYXNlLm9yZy9hdXRoL3YxIiwiYXVkIjoiMCIsIm5iZiI6MTcwMDA1OTI3NiwiaWF0IjoxNzAwMDU5Mjc2LCJqdGkiOiI0MzUyIiwic3ViIjoiMzQ2NjE2OSJ9.FjHPBA-O0GZTl7UDRXjvbL1BEaRei3OghJVaaUP6sd5zeSXF_bYRWnxU9Mi6dCI1BbK4i29UbpuQ8NE-zwnGRC9Nj0vW5i5PFebw0_-B1BZEUfFRuCjI38mFMH3B2JKob_626LoV4DhmExmHWvumbkRnAxLokCrME0G14ZuBO0oSXIT8wVDLELzjC4FibP1XUnPIsn2IUtWNUCqVycEvBg_XV7khWwsvrLl5peRDUr_f6SWnzEsiEJHe43c8KVFfuMZahpmlw9ujtMVn55y4O4Yhe67uGug4Kvp9eoWdT3Rc2M-CXF2MtTUpg2nmYLZ1Gy-CeBZ4V-IAJRMHVbj8YA') # XXX: insert authtoken
    
# # get filenames and SynID from folder for serial download
# walkedPath = synapseutils.walk_functions.walk(syn, 'syn36007626', ['file'])
# fileDict = dict(list(walkedPath)[0][2])
# fileDict = {filekey_rename(k): v for k, v in fileDict.items()} # edit keys to be compatible with renamed files

# # get manifest for whole folder and metadata
# # entities = synapseutils.syncFromSynapse(syn, 'syn36007626', path="D:\\BRAINRISE\\metadata", downloadFile=False)
# # entities = synapseutils.syncFromSynapse(syn, 'syn53222171', path="D:\\BRAINRISE\\metadata", downloadFile=False)

# # get timepoint and age annotations
# manifest = pd.read_csv("D:\\BRAINRISE\\metadata\synapse_metadata_manifest.tsv", sep='\t')
# metadata = pd.read_csv("D:\\BRAINRISE\\metadata\BRAINRISE_1kD_standardized_demographic_data.csv")


def get_tp(filename: str): # requires manifest

    # BR specific get tp:
    # filename = os.path.splitext(frame_check)[0]

    # try: 
    #     synapse_ID = fileDict[filekey_rename(filename).lower()]
    
    #     assessmentAge = manifest.loc[manifest.id == synapse_ID, 'assessmentAge'].iloc[0]  # assessment age in months
       
    #     if assessmentAge < 5:
    #         return 1
    #     if assessmentAge < 10:
    #         return 2
    #     if assessmentAge < 17:
    #         return 3
    #     if assessmentAge < 31:
    #         return 4
    #     if assessmentAge > 31:
    #         return 5
    # except:
    #     return int(re.search(r'\da', filename)[0].strip('a'))

    return re.search(r'timepoint\d+', filename)[0]  # XXX: edit this, re.search(r'\d+a', filename)[0]


def analysis_sequence(frame_check, checks):
    #     print('Data quality check for ' + frame_check)

    # for each csv file in settings.FRAME_CHECKS, generate a quality score from keypoint detections
    data_quality = pd.read_csv(os.path.join(settings.FRAME_CHECKS, frame_check), index_col=0)
    quality_score, _perfect_frames = data_quality_check(data_quality, checks)

    ppt = get_ppt(frame_check) # get participant from filename
    tp = get_tp(frame_check) # get timepoint from filename
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
    results = list(pool.starmap(analysis_sequence, tqdm(args, total=len(args))))

    camera_scores = []
    for i, result in enumerate(results):
        camera_scores.append(result)
        # TODO: save to individual csvs as you go?

    # save camera quality scores for all cameras
    camera_scores = pd.DataFrame(camera_scores, columns=['Filename', 'ppt', 'tp', 'QualityScore'])
    camera_scores.to_csv(settings.ANALYSIS_FOLDER / "model1_camera_scores.csv")
    # print(camera_scores)

    # find the best camera for each session (participant and timepoint)
    best_cams_idx = list(camera_scores.groupby(['ppt', 'tp'])['QualityScore'].idxmax())
    best_cams = camera_scores.iloc[best_cams_idx]

    return best_cams # .Filename


if __name__ == '__main__':
    pass

