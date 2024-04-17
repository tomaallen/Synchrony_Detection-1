import json
import numpy as np
from scipy.io import savemat, loadmat
from _0_data_constants import *


def nan_nonan_dicts(input_keypoint_series, json_save=False):
    nan_dict = {}
    nonan_dict = {}

    for key, data in input_keypoint_series.items():
        print(key)
        print(np.shape(data))
        if key not in ['__header__', '__version__', '__globals__']:
            row_old = np.zeros([3])
            i = 0
            nan_length = 0
            nonan_length = 0
            nan_begin = 0
            nonan_begin = 0
            nan_series_list = []
            nonan_series_list = []
            nonan_begin_value = 0
            # nonan_end_value = 0
            # nonan_series_dict = {}
            for row in data:
                if np.isnan(row[0]):
                    if nan_length == 0:
                        nan_begin = i
                    if nonan_length > 0:
                        nonan_end_value = row_old[0:2]
                        nonan_series_dict = {'begin': nonan_begin, 'length': nonan_length,
                                             'start values': nonan_begin_value.tolist(),
                                             'end values': nonan_end_value.tolist()}
                        nonan_series_list.append(nonan_series_dict)
                        # print(nonan_series_list)
                        # input()
                        if nonan_length <= 5:
                            print("ATTENTION: nonan length", nonan_length, "starting", nonan_begin, "at", key)
                        nonan_length = 0
                    nan_length += 1
                else:
                    if nonan_length == 0:
                        nonan_begin = i
                        nonan_begin_value = row[0:2]
                    if nan_length > 0:
                        nan_series_list.append((nan_begin, nan_length))
                        nan_length = 0
                    nonan_length += 1

                i += 1
                row_old = row
            if nonan_length > 0:
                nonan_end_value = row_old[0:2]
                nonan_series_dict = {'begin': nonan_begin, 'length': nonan_length,
                                     'start values': nonan_begin_value.tolist(),
                                     'end values': nonan_end_value.tolist()}
                nonan_series_list.append(nonan_series_dict)
                if nonan_length <= 5:
                    print("ATTENTION: nonan length", nonan_length, "starting", nonan_begin, "at", key)
                # nonan_length = 0
            if nan_length > 0:
                nan_series_list.append((nan_begin, nan_length))
                # nan_length = 0
            nan_dict[key] = nan_series_list
            nonan_dict[key] = nonan_series_list

    if json_save:
        with open(INPUT_FOLDER + PREFIX + JSON_NAN_LIST, 'w') as fp:
            json.dump(nan_dict, fp)

        with open(INPUT_FOLDER + PREFIX + JSON_NONAN_LIST, 'w') as fp:
            json.dump(nonan_dict, fp)

    return nonan_dict, nan_dict


def remove_short_nonan(input_keypoint_series, nonan_list_dict):
    print('==============Removing short nonan==================')
    for key, data in input_keypoint_series.items():
        print(key)
        print(np.shape(data))
        if key not in ['__header__', '__version__', '__globals__']:
            for nonan_slice in nonan_list_dict[key]:
                if nonan_slice['length'] <= 5:
                    print(nonan_slice)
                    input_keypoint_series[key] \
                        [nonan_slice['begin']:nonan_slice['begin'] + nonan_slice['length'], 0:2] = np.nan


keypoint_matrices_dict_NaN = loadmat(INPUT_FOLDER + PREFIX + MATLAB_PREPROCESSED_DATA_FILE)

nonan_dict, nan_dict = nan_nonan_dicts(keypoint_matrices_dict_NaN)
# with open(FOLDER + JSON_NONAN_LIST, 'r') as f:
#     data = json.load(f)

remove_short_nonan(keypoint_matrices_dict_NaN, nonan_dict)
savemat(INPUT_FOLDER + PREFIX + MATLAB_PREPROCESSED_DATA_FILE_REMOVE_SHORT_NONAN, keypoint_matrices_dict_NaN)
