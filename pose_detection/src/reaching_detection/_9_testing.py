# Python program to write
# text on video

from scipy.io import savemat, loadmat
import numpy as np
import cv2
from _0_data_constants import *
import xgboost as xgb
import matplotlib.pyplot as plt


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
# Add NaN to match dimension for stereo camera input
an_array = np.empty(X_test.shape)
an_array[:] = np.NaN
X_test = np.hstack((an_array, X_test))

print('shape:', X_test.shape)

xg_reg = xgb.XGBRegressor()
xg_reg.load_model(INPUT_FOLDER + XGB_MODEL_NAN)

preds = xg_reg.predict(X_test)

t = np.linspace(1,X_test.shape[0],X_test.shape[0])
plt.plot(t,preds,linestyle="",marker="*")
plt.xlabel('no. of frames')
plt.ylabel('score')
plt.grid(True)

plt.show()
