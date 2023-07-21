import json
from scipy.io import savemat, loadmat

import xgboost as xgb
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from _0_data_constants import *
#
# X_train_1 = loadmat('demo1/Camcorder 1 Demo_reaching_training_input_all_number_mono.mat')['reaching_training_input_data']
# y_train_1 = loadmat('demo1/Camcorder 1 Demo_reaching_training_labels_all_number_mono.mat')['reaching_training_label']
# X_val_1 = loadmat('demo1/Camcorder 1 Demo_reaching_test_input_all_number_mono.mat')['reaching_test_input_data']
# y_val_1 = loadmat('demo1/Camcorder 1 Demo_reaching_test_labels_all_number_mono.mat')['reaching_test_label']
#
X_train = loadmat('demo2/Camcorder 2 Demo_reaching_training_input_all_number_mono.mat')['reaching_training_input_data']
y_train = loadmat('demo2/Camcorder 2 Demo_reaching_training_labels_all_number_mono.mat')['reaching_training_label']
X_val = loadmat('demo2/Camcorder 2 Demo_reaching_test_input_all_number_mono.mat')['reaching_test_input_data']
y_val = loadmat('demo2/Camcorder 2 Demo_reaching_test_labels_all_number_mono.mat')['reaching_test_label']
#
X_test = loadmat('data/lp007/Camcorder 1_Eval room_STT_reaching_test_input_all_number_mono'
                 '.mat')['reaching_test_input_data']
y_test = loadmat('data/lp007/Camcorder 1_Eval room_STT_reaching_test_labels_all_number_mono'
                 '.mat')['reaching_test_label']
#
# print('shape of data:', X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape)
# print('any NaN values:', np.isnan(X_train).any(), np.isnan(y_train).any(), np.isnan(X_val).any(),
#       np.isnan(y_val).any(), np.isnan(X_test).any(), np.isnan(y_test).any())


# first neural network with keras tutorial
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K
from sklearn.metrics import classification_report

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    print(true_positives)
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    print(possible_positives)
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    print(true_positives)
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    print(predicted_positives)
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
# load the dataset
# dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')
# split into input (X) and output (y) variables
X = X_train
y = y_train
# define the keras model
model = Sequential()
model.add(Dense(10, input_shape=(10,), activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(X, y, epochs=20, batch_size=10, validation_data=(X_val, y_val))
# evaluate the keras model
# loss, accuracy, f1_score, precision, recall = model.evaluate(X_val, y_val)
# print('Accuracy: %.2f' % (accuracy*100))

y_train_dense=model.predict(X_train)

t = np.linspace(1,X_train.shape[0],X_train.shape[0])
plt.plot(t,y_train,t,y_train_dense,linestyle="",marker="*")
plt.xlabel('no. of frames')
plt.ylabel('score')
plt.grid(True)

y_test_dense=model.predict(X_test[7500:,:])

print(precision_m(y_test[7500:,:].astype(float), y_test_dense.astype(float)))
print(recall_m(y_test[7500:,:].astype(float), y_test_dense.astype(float)))

t = np.linspace(7500,X_test.shape[0],X_test.shape[0]-7500)
plt.plot(t,y_test[7500:],t,y_test_dense,linestyle="",marker="*")
plt.xlabel('no. of frames')
plt.ylabel('score')
plt.grid(True)

print(classification_report(y_test[7500:].astype(int), np.where(y_test_dense >= 0.5, 1, 0)))

plt.show()