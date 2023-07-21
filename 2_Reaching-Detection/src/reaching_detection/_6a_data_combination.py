import json
from scipy.io import savemat, loadmat

import numpy as np
import matplotlib.pyplot as plt

from _0_data_constants import *

# load Cam 1
folder = 'demo1/'
prefix = 'Camcorder 1 Demo_'
X_train_1 = loadmat(folder + prefix + MATLAB_TRAINING_INPUT_NAN)['reaching_training_input_data']
y_train_1 = loadmat(folder + prefix + MATLAB_TRAINING_OUTPUT_NAN)['reaching_training_label']
X_test_1 = loadmat(folder + prefix + MATLAB_TEST_INPUT_NAN)['reaching_test_input_data']
y_test_1 = loadmat(folder + prefix + MATLAB_TEST_OUTPUT_NAN)['reaching_test_label']

print('shape of data for cam 1:', X_train_1.shape, X_test_1.shape, y_train_1.shape, y_test_1.shape)

# load Cam 2
folder = 'demo2/'
prefix = 'Camcorder 2 Demo_'
X_train_2 = loadmat(folder + prefix + MATLAB_TRAINING_INPUT_NAN)['reaching_training_input_data']
y_train_2 = loadmat(folder + prefix + MATLAB_TRAINING_OUTPUT_NAN)['reaching_training_label']
X_test_2 = loadmat(folder + prefix + MATLAB_TEST_INPUT_NAN)['reaching_test_input_data']
y_test_2 = loadmat(folder + prefix + MATLAB_TEST_OUTPUT_NAN)['reaching_test_label']

print('shape of data for cam 2:', X_train_2.shape, X_test_2.shape, y_train_2.shape, y_test_2.shape)


# Combine 2 cameras
X_train = np.hstack([X_train_1, X_train_2])
X_test = np.hstack([X_test_1, X_test_2])

y_train = np.max(np.hstack([y_train_1, y_train_2]), axis=1, keepdims=True)
y_test = np.max(np.hstack([y_test_1, y_test_2]), axis=1, keepdims=True)


# Only cam 1
# Add NaN to match dimension for stereo camera input
an_array = np.empty(X_train_1.shape)
an_array[:] = np.NaN
X_train_1_only = np.hstack((X_train_1, an_array))
y_train_1_only = y_train_1
X_train = np.vstack((X_train, X_train_1_only))
y_train = np.vstack((y_train, y_train_1_only))

# Only cam 2
# Add NaN to match dimension for stereo camera input
an_array = np.empty(X_train_2.shape)
an_array[:] = np.NaN
X_train_2_only = np.hstack((an_array, X_train_2))
y_train_2_only = y_train_2
X_train = np.vstack((X_train, X_train_2_only))
y_train = np.vstack((y_train, y_train_2_only))

savemat(INPUT_FOLDER + PREFIX + MATLAB_TRAINING_INPUT_NAN_STEREO, {'reaching_training_input_data': X_train})
savemat(INPUT_FOLDER + PREFIX + MATLAB_TRAINING_OUTPUT_NAN_STEREO, {'reaching_training_label': y_train})
savemat(INPUT_FOLDER + PREFIX + MATLAB_TEST_INPUT_NAN_STEREO, {'reaching_test_input_data': X_test})
savemat(INPUT_FOLDER + PREFIX + MATLAB_TEST_OUTPUT_NAN_STEREO, {'reaching_test_label': y_test})

add_nan_to_test = False
if add_nan_to_test:
    X_test[60:70, 12] = np.nan
    X_test[80:90, 12] = np.nan
    savemat(INPUT_FOLDER + PREFIX + MATLAB_TEST_INPUT_NAN_ADD_NAN, {'reaching_test_input_data': X_test})


replace_nan_by_number = True
if replace_nan_by_number:
    X_train = np.nan_to_num(X_train, nan=-5)
    y_train = np.nan_to_num(y_train, nan=0)
    X_test = np.nan_to_num(X_test, nan=-5)
    y_test = np.nan_to_num(y_test, nan=0)
    savemat(INPUT_FOLDER + PREFIX + MATLAB_TRAINING_INPUT_ALL_NUMBER_STEREO, {'reaching_training_input_data': X_train})
    savemat(INPUT_FOLDER + PREFIX + MATLAB_TRAINING_OUTPUT_ALL_NUMBER_STEREO, {'reaching_training_label': y_train})
    savemat(INPUT_FOLDER + PREFIX + MATLAB_TEST_INPUT_ALL_NUMBER_STEREO, {'reaching_test_input_data': X_test})
    savemat(INPUT_FOLDER + PREFIX + MATLAB_TEST_OUTPUT_ALL_NUMBER_STEREO, {'reaching_test_label': y_test})