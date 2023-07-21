import json
from scipy.io import savemat, loadmat

import numpy as np

from _0_data_constants import *

X_train = loadmat(INPUT_FOLDER + PREFIX + MATLAB_TRAINING_INPUT_NAN)['reaching_training_input_data']
y_train = loadmat(INPUT_FOLDER + PREFIX + MATLAB_TRAINING_OUTPUT_NAN)['reaching_training_label']
X_test = loadmat(INPUT_FOLDER + PREFIX + MATLAB_TEST_INPUT_NAN_ADD_NAN)['reaching_test_input_data']
y_test = loadmat(INPUT_FOLDER + PREFIX + MATLAB_TEST_OUTPUT_NAN)['reaching_test_label']

X_train_shape = X_train.shape
X_test_shape = X_test.shape
print('shape of data:', X_train.shape, X_test.shape, y_train.shape, y_test.shape)
print('any NaN values:', np.isnan(X_train).any(), np.isnan(X_test).any(), np.isnan(y_train).any(),
      np.isnan(y_test).any())
# input()

time_kernel = 50
time_stride = 25

no_windows = (X_train_shape[0] - time_kernel) // time_stride + 1
print('no_windows is', no_windows)

# training input for LSTM
X_train_lstm = np.zeros([no_windows, time_kernel, X_test_shape[1]])

for i in range(no_windows):
    x = X_train[i * time_stride:time_kernel + i * time_stride, :]  # np.reshape(,[-1,X_train_shape[1]])
    # print(np.shape(X_train_lstm[i:i+1,:,:]), np.shape(x))
    X_train_lstm[i:i + 1, :, :] = x

# training output for LSTM
y_train_lstm = np.zeros([no_windows, 1])

for i in range(no_windows):
    y = y_train[i * time_stride:time_kernel + i * time_stride, :]  # np.reshape(,[-1,X_train_shape[1]])
    # print(y)
    # print(np.mean(y))
    if np.mean(y) >= 2/time_kernel:
        y = 1
        y_train_lstm[i:i + 1, :] = y

# print(X_train_lstm[-1, :, :])
print(y_train_lstm)
