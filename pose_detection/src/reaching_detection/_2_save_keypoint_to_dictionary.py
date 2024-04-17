import json
import numpy as np
from scipy.io import savemat
from _0_data_constants import *

with open(INPUT_FOLDER + PREFIX + JSONL_FILE, 'r') as f:
    data = json.load(f)

# Output: {'name': 'Bob', 'languages': ['English', 'French']}
# list_of_keypoints = ['Nose', 'Neck', 'RShoulder', 'RElbow', 'RWrist', 'LShoulder', 'LElbow', 'LWrist',
#                      'MidHip', 'RHip', 'LHip']
list_of_keypoints = ["Nose", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow", "LWrist",
                     "MidHip", "RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle", "REye", "LEye", "REar",
                     "LEar", "LBigToe", "LSmallToe", "LHeel", "RBigToe", "RSmallToe", "RHeel"]
list_of_matrices = [np.zeros([NO_CSV_FILES, 3])] * len(list_of_keypoints)
for i, matrix in enumerate(list_of_matrices):
    list_of_matrices[i] = np.zeros([NO_CSV_FILES, 3])
for i, matrix in enumerate(list_of_matrices):
    list_of_matrices[i][:] = np.nan
# print(list_of_matrices)

# print(keypoint_matrix_dict)
# input()
for i, keypoint in enumerate(list_of_keypoints):
    for key, pose in data.items():
        try:
            # print(key, pose['Data']['74'])
            # print(np.asarray(pose['Data']['18'][keypoint]))
            # key_ = int(key) - 6310
            list_of_matrices[i][int(key) - 1:int(key)] = np.asarray(pose['Data']['2'][keypoint])
            # if int(key)<=9000:
            #     list_of_matrices[i][int(key) - 1:int(key)] = np.asarray(pose['Data']['2'][keypoint])
            # elif int(key)>=9830:
            #     list_of_matrices[i][int(key) - 1:int(key)] = np.asarray(pose['Data']['54'][keypoint])
            # print(pose['Data']['74'][keypoint])
            # input()
        except Exception as e:
            print(e)
# print(data)
keypoint_matrix_dict = dict(zip(list_of_keypoints, list_of_matrices))
savemat(INPUT_FOLDER + PREFIX + MATLAB_RAW_DATA_FILE, keypoint_matrix_dict)
