# Python program to write
# text on video

from scipy.io import savemat, loadmat
import numpy as np
import cv2
from _0_data_constants import *
import xgboost as xgb
import matplotlib.pyplot as plt


stereo = False

angles_dict = loadmat(INPUT_FOLDER + PREFIX + MATLAB_ANGLES)
input_data = angles_dict['Nose_Neck_LShoulder'][:, 0:1]
# print(input_data)
print('data dimension:', input_data.shape)

for key, data in angles_dict.items():
    # print(key)
    # print(np.shape(data))
    if key not in ['__header__', '__version__', '__globals__', 'Nose_Neck_LShoulder']:
        # print(input_data.shape, data.shape)
        input_data = np.hstack((input_data, data[:, 0:1]))


X_test = input_data
X_test_mono = input_data

# Add NaN to match dimension for stereo camera input
if stereo:
    an_array = np.empty(X_test.shape)
    an_array[:] = np.NaN
    X_test = np.hstack((an_array, X_test))

print('shape of X:', X_test.shape)
list_of_pointing = []

# LP007
# list_of_reaching = [[1200, 1410], [1720, 1780]]

# LP006
# list_of_pointing = []
list_of_reaching = [[1000, 1037], [1070, 1130], [1238, 1260], [1388, 1392], [1489, 1495], [1547, 1585], [1680, 1700]]

# C003
# list_of_reaching = []

# C005
# list_of_reaching = [[450, 520], [4790, 5020], [5250, 5620], [5930, 6760], [6930, 7050], [7800, 8260],
#                     [11380,11870], [12090,12420], [13030,13400], [14520,14910], [15000,15080], [15370,15570]]

# C007
# list_of_reaching = [[1930,2240], [2680,3420], [4230,4830], [4975,6400], [6610,7600], [7820,8680], [8736,9320],
#                     [9440,10000], [10230,11100]]

y_test = np.zeros([input_data.shape[0], 1])

for range_ in list_of_reaching:
    y_test[range_[0]:range_[1] + 1, 0] = 1
for range_ in list_of_pointing:
    y_test[range_[0]:range_[1] + 1, 0] = 1

print('shape of y:', y_test.shape)

training_data = True
if training_data:
    for i, row in enumerate(input_data):
        # if i > 480:
        #     print(row)
        #     input()
        if all(~np.isnan(row)): #~np.isnan(row[3]) and ~np.isnan(row[4]) and ~np.isnan(row[5]): #
            pass
        else:
            if any(range_[0] <= i <= range_[1] for range_ in list_of_reaching):
                # print([range_[0] <= i <= range_[1] for range_ in list_of_reaching])
                print('Note in labeling:', i)
                # input()
            y_test[i, 0] = 0  # zero labels for any NaN in input
            # will use all data (include NaN) for training

y_test_mono = y_test

if stereo:
    savemat(INPUT_FOLDER + PREFIX + MATLAB_TEST_INPUT_NAN_STEREO, {'reaching_test_input_data': X_test})
    savemat(INPUT_FOLDER + PREFIX + MATLAB_TEST_OUTPUT_NAN_STEREO, {'reaching_test_label': y_test})

savemat(INPUT_FOLDER + PREFIX + MATLAB_TEST_INPUT_NAN_MONO, {'reaching_test_input_data': X_test_mono})
savemat(INPUT_FOLDER + PREFIX + MATLAB_TEST_OUTPUT_NAN_MONO, {'reaching_test_label': y_test_mono})

replace_nan_by_number = True
if replace_nan_by_number:
    if stereo:
        X_test = np.nan_to_num(X_test, nan=-5)
        y_test = np.nan_to_num(y_test, nan=0)
        savemat(INPUT_FOLDER + PREFIX + MATLAB_TEST_INPUT_ALL_NUMBER_STEREO, {'reaching_test_input_data': X_test})
        savemat(INPUT_FOLDER + PREFIX + MATLAB_TEST_OUTPUT_ALL_NUMBER_STEREO, {'reaching_test_label': y_test})

    X_test_mono = np.nan_to_num(X_test_mono, nan=-5)
    y_test_mono = np.nan_to_num(y_test_mono, nan=0)
    savemat(INPUT_FOLDER + PREFIX + MATLAB_TEST_INPUT_ALL_NUMBER_MONO, {'reaching_test_input_data': X_test_mono})
    savemat(INPUT_FOLDER + PREFIX + MATLAB_TEST_OUTPUT_ALL_NUMBER_MONO, {'reaching_test_label': y_test_mono})