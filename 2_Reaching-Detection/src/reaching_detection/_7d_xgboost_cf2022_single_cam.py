import json
from scipy.io import savemat, loadmat

import xgboost as xgb
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report


from _0_data_constants import *

X_train = loadmat('..\\data\\CF 2022\\training_data.mat')['training_input_nan']
y_train = loadmat('..\\data\\CF 2022\\training_data.mat')['training_label_nan']
X_val = loadmat('..\\data\\CF 2022\\val_data.mat')['val_input_nan']
y_val = loadmat('..\\data\\CF 2022\\val_data.mat')['val_label_nan']

X_test = loadmat('..\\data\\CF 2022\\test_data.mat')['test_input_nan']
y_test = loadmat('..\\data\\CF 2022\\test_data.mat')['test_label_nan']

print('shape of data:', X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape)
print('any NaN values:', np.isnan(X_train).any(), np.isnan(y_train).any(), np.isnan(X_val).any(),
      np.isnan(y_val).any(), np.isnan(X_test).any(), np.isnan(y_test).any())# input()

xg_reg = xgb.XGBRegressor(objective='reg:linear', colsample_bytree=0.8, learning_rate=0.1,
                          max_depth=3, alpha=0.5, n_estimators=1000)

xg_reg.fit(X_train, y_train)

print(xg_reg.objective)

preds_val = xg_reg.predict(X_val)

preds = xg_reg.predict(X_test)
# print(np.shape(preds_train), np.shape(preds))

# total_preds = np.vstack((np.expand_dims(preds_train,axis=1),np.expand_dims(preds,axis=1)))
# print('total', np.shape(total_preds))
# with open('total_pred.npy', 'wb') as f:
# np.save(FOLDER + XGB_MODEL, total_preds)
xg_reg.save_model(INPUT_FOLDER + PREFIX + XGB_MODEL_NAN)

rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))

plt.figure()
t = np.linspace(1, X_val.shape[0], X_val.shape[0])
plt.plot(t, y_val, t, preds_val, linestyle="", marker="*")
plt.xlabel('no. of frames')
plt.ylabel('score')
plt.grid(True)

plt.figure()
t = np.linspace(1, X_test.shape[0], X_test.shape[0])
plt.plot(t, y_test, t, preds, linestyle="", marker="o")
plt.xlabel('no. of frames')
plt.ylabel('score')
plt.grid(True)
# plt.show()

xgb.plot_importance(xg_reg)
# plt.rcParams['figure.figsize'] = [5, 5]



y_val_pred=xg_reg.predict(X_val)
print('valuation')
print(classification_report(y_val.astype(int), np.where(y_val_pred > 0.4, 1, 0)))
# print(classification_report(y_val.astype(int), np.round(y_val_pred.astype(float))))

y_test_pred=xg_reg.predict(X_test)
print('test')
print(classification_report(y_test.astype(int), np.where(y_test_pred > 0.4, 1, 0)))
# print(classification_report(y_test.astype(int), y_test_pred.astype(float)))


plt.show()
