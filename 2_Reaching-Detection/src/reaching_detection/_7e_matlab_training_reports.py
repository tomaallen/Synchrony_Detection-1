from scipy.io import savemat, loadmat

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report


y_val = loadmat('..\\data\\CF 2022\\train_by_GP_2022_08_10_val_test_results.mat')['Y_val_truth']
y_val_pred = loadmat('..\\data\\CF 2022\\train_by_GP_2022_08_10_val_test_results.mat')['Y_val']
y_test = loadmat('..\\data\\CF 2022\\train_by_GP_2022_08_10_val_test_results.mat')['Y_test_truth']
y_test_pred = loadmat('..\\data\\CF 2022\\train_by_GP_2022_08_10_val_test_results.mat')['Y_test']


print('valuation')
print(classification_report(y_val.astype(int), np.where(y_val_pred > 0.4, 1, 0)))
# print(classification_report(y_val.astype(int), np.round(y_val_pred.astype(float))))

print('test')
print(classification_report(y_test.astype(int), np.where(y_test_pred > 0.4, 1, 0)))

plt.figure()
t = np.linspace(1, y_val.shape[0], y_val.shape[0])
plt.plot(t, y_val, t, y_val_pred, linestyle="", marker="*")
plt.xlabel('no. of frames')
plt.ylabel('score')
plt.grid(True)

plt.figure()
t = np.linspace(1, y_test.shape[0], y_test.shape[0])
plt.plot(t, y_test, t, y_test_pred, linestyle="", marker="o")
plt.xlabel('no. of frames')
plt.ylabel('score')
plt.grid(True)

plt.show()