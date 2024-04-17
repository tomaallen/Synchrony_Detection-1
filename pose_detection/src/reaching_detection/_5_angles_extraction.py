import json
import numpy as np
from scipy.io import savemat, loadmat
from _0_data_constants import *

keypoint_matrices_dict = loadmat(INPUT_FOLDER + PREFIX + MATLAB_PREPROCESSED_DATA_FILE_REMOVE_SHORT_NONAN)

angles_dict = {}

vector_dict = {}


# Vectors
def vector_confidence_calc(points_1, points_2):
    return np.c_[points_2[:, 0:2] - points_1[:, 0:2], np.amin(np.c_[points_2[:, 2:3], points_1[:, 2:3]], axis=1)]


# Nose2Neck 'Nose', 'Neck'
Nose2Neck = vector_confidence_calc(keypoint_matrices_dict['Nose'], keypoint_matrices_dict['Neck'])
vector_dict['Nose2Neck'] = Nose2Neck

# Neck2LShoulder 'Neck', 'LShoulder'
Neck2LShoulder = vector_confidence_calc(keypoint_matrices_dict['Neck'], keypoint_matrices_dict['LShoulder'])
vector_dict['Neck2LShoulder'] = Neck2LShoulder

# Neck2RShoulder 'Neck', 'RShoulder'
Neck2RShoulder = vector_confidence_calc(keypoint_matrices_dict['Neck'], keypoint_matrices_dict['RShoulder'])
vector_dict['Neck2RShoulder'] = Neck2RShoulder

# LShoulder2LElbow 'LShoulder', 'LElbow'
LShoulder2LElbow = vector_confidence_calc(keypoint_matrices_dict['LShoulder'], keypoint_matrices_dict['LElbow'])
vector_dict['LShoulder2LElbow'] = LShoulder2LElbow

# LElbow2LWrist 'LShoulder', 'LWrist'
LElbow2LWrist = vector_confidence_calc(keypoint_matrices_dict['LElbow'], keypoint_matrices_dict['LWrist'])
vector_dict['LElbow2LWrist'] = LElbow2LWrist

# RShoulder2LElbow 'RShoulder', 'RElbow'
RShoulder2RElbow = vector_confidence_calc(keypoint_matrices_dict['RShoulder'], keypoint_matrices_dict['RElbow'])
vector_dict['RShoulder2RElbow'] = RShoulder2RElbow

# RElbow2RWrist 'RShoulder', 'RWrist'
RElbow2RWrist = vector_confidence_calc(keypoint_matrices_dict['RElbow'], keypoint_matrices_dict['RWrist'])
vector_dict['RElbow2RWrist'] = RElbow2RWrist

# Neck2MidHip 'Neck', 'MidHip'
Neck2MidHip = vector_confidence_calc(keypoint_matrices_dict['Neck'], keypoint_matrices_dict['MidHip'])
vector_dict['Neck2MidHip'] = Neck2MidHip

# MidHip2RHip 'MidHip', 'RHip'
MidHip2RHip = vector_confidence_calc(keypoint_matrices_dict['MidHip'], keypoint_matrices_dict['RHip'])
vector_dict['MidHip2RHip'] = MidHip2RHip

# MidHip2RHip 'MidHip', 'LHip'
MidHip2LHip = vector_confidence_calc(keypoint_matrices_dict['MidHip'], keypoint_matrices_dict['LHip'])
vector_dict['MidHip2LHip'] = MidHip2LHip

savemat(INPUT_FOLDER + PREFIX + MATLAB_VECTORS, vector_dict)


# Angles
def angles_confidence_calc(vectors_1, vectors_2):
    v1 = vectors_1[:, 0:2]
    v2 = vectors_2[:, 0:2]
    # print('v1v2 is', v1, v2)
    unit_vector_1 = v1 / np.linalg.norm(v1, axis=1)[:, None]
    unit_vector_2 = v2 / np.linalg.norm(v2, axis=1)[:, None]
    dot_product = np.sum(unit_vector_1 * unit_vector_2, axis=1)
    angle_radian = np.arccos(dot_product)
    angle_degree = np.degrees(angle_radian)
    # print('angle is', angle_radian, np.c_[vectors_2[:, 2:3],vectors_1[:, 2:3]])
    # print('min í', np.amin(np.c_[vectors_2[:, 2:3], vectors_1[:, 2:3]], axis=1))
    # input()
    # print(np.c_[angle_radian, np.amin(np.c_[vectors_2[:, 2:3], vectors_1[:, 2:3]], axis=1)])
    return np.c_[angle_radian, np.amin(np.c_[vectors_2[:, 2:3], vectors_1[:, 2:3]], axis=1)]


# 0: Nose_Neck_LShoulder
print("++++++++++++++++++++++++++++++++++++++++++++++++++")
Nose_Neck_LShoulder = angles_confidence_calc(Nose2Neck, Neck2LShoulder)
# Nose_Neck_LShoulder = np.nan_to_num(Nose_Neck_LShoulder)
angles_dict['Nose_Neck_LShoulder'] = Nose_Neck_LShoulder


# 1: Neck_LShoulder_LElbow
print("++++++++++++++++++++++++++++++++++++++++++++++++++")
Neck_LShoulder_LElbow = angles_confidence_calc(Neck2LShoulder, LShoulder2LElbow)
# Neck_LShoulder_LElbow = np.nan_to_num(Neck_LShoulder_LElbow)
angles_dict['Neck_LShoulder_LElbow'] = Neck_LShoulder_LElbow

# 2: LShoulder_LElbow_LWrist
print("++++++++++++++++++++++++++++++++++++++++++++++++++")
LShoulder_LElbow_LWrist = angles_confidence_calc(LShoulder2LElbow, LElbow2LWrist)
# LShoulder_LElbow_LWrist = np.nan_to_num(LShoulder_LElbow_LWrist)
angles_dict['LShoulder_LElbow_LWrist'] = LShoulder_LElbow_LWrist

# 3: Nose_Neck_RShoulder
print("++++++++++++++++++++++++++++++++++++++++++++++++++")
Nose_Neck_RShoulder = angles_confidence_calc(Nose2Neck, Neck2RShoulder)
# Nose_Neck_RShoulder = np.nan_to_num(Nose_Neck_RShoulder)
angles_dict['Nose_Neck_RShoulder'] = Nose_Neck_RShoulder

# 4: Neck_RShoulder_RElbow
print("++++++++++++++++++++++++++++++++++++++++++++++++++")
Neck_RShoulder_RElbow = angles_confidence_calc(Neck2RShoulder, RShoulder2RElbow)
# Neck_RShoulder_RElbow = np.nan_to_num(Neck_RShoulder_RElbow)
angles_dict['Neck_RShoulder_RElbow'] = Neck_RShoulder_RElbow

# 5: RShoulder_RElbow_RWrist
print("++++++++++++++++++++++++++++++++++++++++++++++++++")
RShoulder_RElbow_RWrist = angles_confidence_calc(RShoulder2RElbow, RElbow2RWrist)
# RShoulder_RElbow_RWrist = np.nan_to_num(RShoulder_RElbow_RWrist)
angles_dict['RShoulder_RElbow_RWrist'] = RShoulder_RElbow_RWrist

# 6: LShoulder_Neck_MidHip
print("++++++++++++++++++++++++++++++++++++++++++++++++++")
LShoulder_Neck_MidHip = angles_confidence_calc(Neck2LShoulder, Neck2MidHip)
# LShoulder_Neck_MidHip = np.nan_to_num(LShoulder_Neck_MidHip)
angles_dict['LShoulder_Neck_MidHip'] = LShoulder_Neck_MidHip

# 7: RShoulder_Neck_MidHip
print("++++++++++++++++++++++++++++++++++++++++++++++++++")
RShoulder_Neck_MidHip = angles_confidence_calc(Neck2RShoulder, Neck2MidHip)
# RShoulder_Neck_MidHip = np.nan_to_num(RShoulder_Neck_MidHip)
angles_dict['RShoulder_Neck_MidHip'] = RShoulder_Neck_MidHip

# 8: Neck_MidHip_LHip
print("++++++++++++++++++++++++++++++++++++++++++++++++++")
Neck_MidHip_LHip = angles_confidence_calc(Neck2MidHip, MidHip2LHip)
# Neck_MidHip_LHip = np.nan_to_num(Neck_MidHip_LHip)
angles_dict['Neck_MidHip_LHip'] = Neck_MidHip_LHip

# 9: Neck_MidHip_RHip
print("++++++++++++++++++++++++++++++++++++++++++++++++++")
Neck_MidHip_RHip = angles_confidence_calc(Neck2MidHip, MidHip2RHip)
# Neck_MidHip_RHip = np.nan_to_num(Neck_MidHip_RHip)
angles_dict['Neck_MidHip_RHip'] = Neck_MidHip_RHip


savemat(INPUT_FOLDER + PREFIX + MATLAB_ANGLES, angles_dict)
