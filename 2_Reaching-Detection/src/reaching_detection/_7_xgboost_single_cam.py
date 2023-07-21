import json
from scipy.io import savemat, loadmat

import xgboost as xgb
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from _0_data_constants import *

X_train = loadmat(INPUT_FOLDER + PREFIX + MATLAB_TRAINING_INPUT_NAN)['reaching_training_input_data']
y_train = loadmat(INPUT_FOLDER + PREFIX + MATLAB_TRAINING_OUTPUT_NAN)['reaching_training_label']
X_test = loadmat(INPUT_FOLDER + PREFIX + MATLAB_TEST_INPUT_NAN)['reaching_test_input_data']
y_test = loadmat(INPUT_FOLDER + PREFIX + MATLAB_TEST_OUTPUT_NAN)['reaching_test_label']

print('shape of data:', X_train.shape, X_test.shape, y_train.shape, y_test.shape)
print('any NaN values:', np.isnan(X_train).any(), np.isnan(X_test).any(),np.isnan(y_train).any(),np.isnan(y_test).any())
# input()

xg_reg = xgb.XGBRegressor(objective='reg:linear', colsample_bytree=0.9, learning_rate=0.1,
                          max_depth=10, alpha=0.5, n_estimators=500)

xg_reg.fit(X_train,y_train)

print(xg_reg.objective)

preds_train = xg_reg.predict(X_train)

preds = xg_reg.predict(X_test)
# print(np.shape(preds_train), np.shape(preds))

# total_preds = np.vstack((np.expand_dims(preds_train,axis=1),np.expand_dims(preds,axis=1)))
# print('total', np.shape(total_preds))
# with open('total_pred.npy', 'wb') as f:
# np.save(FOLDER + XGB_MODEL, total_preds)
xg_reg.save_model(INPUT_FOLDER + PREFIX + XGB_MODEL_NAN)

rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))

t = np.linspace(1,X_train.shape[0],X_train.shape[0])
plt.plot(t,y_train,t,preds_train,linestyle="",marker="*")
plt.xlabel('no. of frames')
plt.ylabel('score')
plt.grid(True)

t = np.linspace(X_train.shape[0]+1,X_train.shape[0]+X_test.shape[0],X_test.shape[0])
plt.plot(t,y_test,t,preds,linestyle="",marker="o")
# plt.show()

xgb.plot_importance(xg_reg)
# plt.rcParams['figure.figsize'] = [5, 5]
plt.show()
