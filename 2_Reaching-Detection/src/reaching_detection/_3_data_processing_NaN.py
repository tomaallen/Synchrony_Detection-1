import json
import numpy as np
from scipy.io import savemat, loadmat
from _0_data_constants import *

keypoint_matrices_dict = loadmat(INPUT_FOLDER + PREFIX + MATLAB_RAW_DATA_FILE)

for key, data in keypoint_matrices_dict.items():
    print(key)
    print(np.shape(data))
    if key not in ['__header__', '__version__', '__globals__']:
        row_old = np.zeros([3])
        i = 0
        k = 1
        h = 1
        for row in data:
            # print(row, row_old)
            # if np.abs(row[0] - row_old[0]) > 30:
            #     k += 1
            # if row[2] == 0 and k == 1:
            #     k += 1
            # if key == "LWrist" and i > 1120:
            #     print('here 0', i, row[0], row[1], row[2])
            #     print('here 1', i, k, row[2], row[0]-row_old[0], row[1] - row_old[1])
            #     print('here 2', k, h,  np.sqrt(h) * 30)
            #     input()
            if i <= 0:
                row_old = row
                row[0] = np.nan
                row[1] = np.nan
            elif (row[2] == 0 or np.abs(row[0] - row_old[0]) > np.sqrt(h) * 30
                  or np.abs(row[1] - row_old[1]) > np.sqrt(h) * 30):
                # if key == "LWrist" and i > 1120:
                #     print('heerererere')
                #     print('row old', row_old)
                #     input()
                # print(row[0], row_old[0])
                row[0] = np.nan
                row[1] = np.nan
                if k < 6:
                    k += 1
                    h = k
                elif k < 15:
                    k += 1
                    h = k*2
                elif k < 30:
                    k += 1
                    h = 50
            else:
                row_old = row
                k = 1
                h = 1
                # input()

            # if key == "LWrist" and i > 494:
            #     print(row, row_old)
            #     input()

            keypoint_matrices_dict[key][i:i + 1, :] = row
            i += 1

savemat(INPUT_FOLDER + PREFIX + MATLAB_PREPROCESSED_DATA_FILE, keypoint_matrices_dict)
