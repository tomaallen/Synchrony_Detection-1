import json
import numpy as np
from scipy.io import savemat, loadmat

import xgboost as xgb
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
from _0_data_constants import *

angles_dict = loadmat(INPUT_FOLDER + PREFIX + MATLAB_ANGLES)
input_data = angles_dict['Nose_Neck_LShoulder'][:, 0:1]
# print(input_data)
print('data dimension:', input_data.shape)

list_of_pointing = [[1388, 1392], [1489, 1495]]
list_of_reaching = [[1000, 1037], [1070, 1130], [1238, 1260], [1547, 1585], [1680, 1700]]

for key, data in angles_dict.items():
    # print(key)
    # print(np.shape(data))
    if key not in ['__header__', '__version__', '__globals__', 'Nose_Neck_LShoulder']:
        # print(input_data.shape, data.shape)
        input_data = np.hstack((input_data, data[:, 0:1]))

output_data = np.zeros([input_data.shape[0], 1])

for range_ in list_of_reaching:
    output_data[range_[0]:range_[1] + 1, 0] = 1

for range_ in list_of_pointing:
    output_data[range_[0]:range_[1] + 1, 0] = 1

nan_index_list = []
#
#
# making NaN (or 0) labels if one feature is NaN
train_with_nan = True
for i, row in enumerate(input_data):
    # if i > 480:
    #     print(row)
    if all(~np.isnan(row)):
        pass
    else:
        if train_with_nan:
            output_data[i, 0] = 0  # zero labels for any NaN in input
            # will use all data (include NaN) for training
        else:
            output_data[i, 0] = np.nan
        nan_index_list.append(i)
    # input()

training_input = input_data[0:1500, :]
training_output = output_data[0:1500, :]
test_input = input_data[1500:, :]
test_output = output_data[1500:, :]
if not train_with_nan:
    nan_index_list_training = [i for i in nan_index_list if i < 1500]
    nan_index_list_test = [i - 1500 for i in nan_index_list if i >= 1500]
    training_input = np.delete(training_input, nan_index_list_training, axis=0)
    training_output = np.delete(training_output, nan_index_list_training, axis=0)
    test_input = np.delete(test_input, nan_index_list_test, axis=0)
    test_output = np.delete(test_output, nan_index_list_test, axis=0)

print('input data dimension:', training_input.shape, test_input.shape)
print('output data dimension:', training_output.shape, test_output.shape)
savemat(INPUT_FOLDER + PREFIX + MATLAB_TRAINING_INPUT_NAN_MONO, {'reaching_training_input_data': training_input})
savemat(INPUT_FOLDER + PREFIX + MATLAB_TRAINING_OUTPUT_NAN_MONO, {'reaching_training_label': training_output})
savemat(INPUT_FOLDER + PREFIX + MATLAB_TEST_INPUT_NAN_MONO, {'reaching_test_input_data': test_input})
savemat(INPUT_FOLDER + PREFIX + MATLAB_TEST_OUTPUT_NAN_MONO, {'reaching_test_label': test_output})

replace_nan_by_number = True
if replace_nan_by_number:
    training_input = np.nan_to_num(training_input, nan=-5)
    training_output = np.nan_to_num(training_output, nan=0)
    test_input = np.nan_to_num(test_input, nan=-5)
    test_output = np.nan_to_num(test_output, nan=0)
    savemat(INPUT_FOLDER + PREFIX + MATLAB_TRAINING_INPUT_ALL_NUMBER_MONO, {'reaching_training_input_data': training_input})
    savemat(INPUT_FOLDER + PREFIX + MATLAB_TRAINING_OUTPUT_ALL_NUMBER_MONO, {'reaching_training_label': training_output})
    savemat(INPUT_FOLDER + PREFIX + MATLAB_TEST_INPUT_ALL_NUMBER_MONO, {'reaching_test_input_data': test_input})
    savemat(INPUT_FOLDER + PREFIX + MATLAB_TEST_OUTPUT_ALL_NUMBER_MONO, {'reaching_test_label': test_output})
