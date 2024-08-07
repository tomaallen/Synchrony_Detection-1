import copy
import os
import sys
import constants
import numpy as np
import json
from scipy.signal import savgol_filter
import pandas as pd
from scipy.optimize import linear_sum_assignment


# functions related to directories (folders) and files
# Structure: (directories start with -, files start with +)
# Parent Dir
#  - Video Dir 1
#     - csv_files
#     - json_files
#     - output_videos
#       + vid.avi
#     - video_info
#     - ...
#  - Video Dir 2
#  - ...

def update_prefix():
    constants.ORIGINAL_VIDEO_NAME = os.path.basename(
        constants.INPUT_FOLDER)
    constants.PREFIX = os.path.basename(constants.INPUT_FOLDER) + '-'

    # "D:\\Temp C\\Users\\Home - Jupiter\\Desktop\\Brazil\\029_15_07_2022_C2"
    # 'Temp Data\\LP016_PCI' # "C:\\Users\\Home - Jupiter\\Desktop\\coordination\\LP031_PCI"  #
    constants.YOLO_INPUT_FOLDER = constants.INPUT_FOLDER
    # OUTPUT_FOLDER is normally the same "Parent dir"
    constants.OUTPUT_FOLDER = constants.INPUT_FOLDER
    # "C:\\Users\\Home - Jupiter\\Desktop\\DATA_HUB\\output_files_leap\\LP012_SyncCam_STT_Demo"

    constants.ORIGINAL_VIDEO_NAME = os.path.basename(constants.INPUT_FOLDER)
    # 'Camcorder 1_Eval room_STT.avi'  #  Note: include the
    # file extension #
    constants.PREFIX = os.path.basename(constants.INPUT_FOLDER) + '-'
    # ORIGINAL_VIDEO_NAME[:ORIGINAL_VIDEO_NAME.rindex('.')] + '_'


def read_video_info(input_folder):
    list_of_video_info_files = file_path_giving_folder("video_info",  parent_dir=input_folder)
    for _file in list_of_video_info_files:
        with open(_file) as file:
            lines = file.readlines()
            lines = [line.rstrip() for line in lines]
            for _line in lines:
                if 'Width' in _line:
                    frame_width = int(_line[6:])
                if 'Height' in _line:
                    frame_height = int(_line[7:])
                if 'FPS' in _line:
                    frame_fps = float(_line[4:])
                if 'Total no. of frames' in _line:
                    total_no_frames = float(_line[20:])
    # print(frame_width, frame_height, frame_fps)
    return frame_width, frame_height, frame_fps, total_no_frames


def read_baby_ids(input_folder):
    list_of_video_info_files = file_path_giving_folder("output_videos", parent_dir=input_folder)
    for _file in list_of_video_info_files:
        if os.path.basename(_file) == 'baby_ids.txt':
            with open(_file) as file:
                texts = file.read()
                id_list = texts.split(",")
                id_list = [[int(_id) for _id in list(filter(None, _id_short.split(" ")))] for _id_short in id_list]
                # id_list_int = [int(_id) for _id in id_list]
            return id_list
    return None


def read_offset(input_folder):
    list_of_video_info_files = file_path_giving_folder("excel_file", parent_dir=input_folder)
    for _file in list_of_video_info_files:
        if os.path.basename(_file) == 'offset.txt':
            with open(_file) as file:
                texts = file.read()
                offset = int(texts) / 1000
            return offset
    return None


def excel_file_path(input_folder):
    list_of_video_info_files = file_path_giving_folder("excel_file", parent_dir=input_folder)
    if len(list_of_video_info_files) == 0:
        return None
    for _file in list_of_video_info_files:
        _, file_extension = os.path.splitext(_file)
        if file_extension == '.xlsx':
            return _file
    return None


def file_path_giving_folder(directory="output_videos", parent_dir=None):
    if parent_dir is None:
        parent_dir = constants.INPUT_FOLDER
    json_folder = os.path.join(parent_dir, directory)
    if not os.path.exists(json_folder):
        print('The folder', str(json_folder), 'does not exists!')
        return None
    else:
        only_files = [f for f in os.listdir(json_folder) if os.path.isfile(os.path.join(json_folder, f))]
        # if only_files.__len__() == 0:
        #     print('There are no files in current folder,', directory)
        list_of_file_path = []
        for _file in only_files:
            list_of_file_path.append(os.path.join(json_folder, _file))

    return sorted(list_of_file_path)


class Polar2D:
    def __init__(self, angle, distance, pole, horizontal=1, conf=0., name=''):
        self.angle = angle
        self.distance = distance
        self.pole = pole
        self.horizontal = horizontal
        self.conf = conf
        self.name = name

    def __str__(self):
        return "angle: " + str(self.angle) \
               + " (" + str(self.angle / np.pi * 180) \
               + "), distance: " + str(self.distance)


def point2d_to_polar2d(point2d, pole2d, horizontal=1):
    conf = np.min([point2d.conf, pole2d.conf], axis=0)
    name = point2d.name + '_' + pole2d.name
    # relative_xy = Point2D(point2d.x - pole2d.x, point2d.y - pole2d.y)
    if horizontal:
        angle = np.arctan((point2d.y - pole2d.y) / (point2d.x - pole2d.x))
    else:
        angle = np.arctan((point2d.x - pole2d.x) / (point2d.y - pole2d.y))
    distance = LineSegment2D(point2d, pole2d).length()
    return Polar2D(angle, distance, pole2d, horizontal, conf, name)


def point2d_to_relative(point2d, origin2d):
    conf = np.min([point2d.conf, origin2d.conf], axis=0)
    name = point2d.name + '_' + origin2d.name
    # relative_xy = Point2D(point2d.x - pole2d.x, point2d.y - pole2d.y)
    x_ = np.round(point2d.x - origin2d.x, 3)
    y_ = np.round(point2d.y - origin2d.y, 3)
    return Point2DRelative(x_, y_, origin2d, conf, name)


class Point2D:
    def __init__(self, x, y, conf=0., name=''):
        self.x = x
        self.y = y
        self.conf = conf
        self.name = name

    def class_dict(self):
        return [self.x, self.y, self.conf]

    def named(self, point_name):
        self.name = point_name

    def is_false_origin(self):
        return self.x == 0 and self.y == 0 and self.conf == 0

    def is_false_origin_approx(self):
        return np.absolute(self.x) <= 10 ** (-8) and np.absolute(self.y) <= 10 ** (-8) and self.conf == 0

    def to_zero(self):
        return Point2D(0., 0., name=self.name)

    def to_nan(self):
        return Point2D(np.nan, np.nan, name=self.name)

    def is_nan(self):
        return np.isnan(self.x) or np.isnan(self.y)

    def save_into_mat(self):
        mat_dict = {self.name: np.c_([self.x, self.y, self.conf])}
        return mat_dict

    def save_into_array(self):
        return np.c_[self.x, self.y, self.conf]

    def __str__(self):
        return "x: " + str(self.x) \
               + ", y: " + str(self.y) \
               + ", name: " + str(self.name)


class Point2DRelative:
    def __init__(self, x, y, origin=Point2D(0., 0.), conf=0., name=''):
        self.x = x
        self.y = y
        self.origin = origin
        self.conf = conf
        self.name = name

    def class_dict(self):
        return [self.x, self.y, self.conf]

    def __str__(self):
        return "x: " + str(self.x) \
               + ", y: " + str(self.y) \
               + ", origin: (" + str(self.origin) \
               + "), name: " + str(self.name)

    def is_false_origin(self):
        return self.x == 0 and self.y == 0 and self.conf == 0

    def is_false_origin_approx(self):
        return np.absolute(self.x) <= 10 ** (-8) and np.absolute(self.y) <= 10 ** (-8) and self.conf == 0

    def to_zero(self):
        return Point2DRelative(0., 0., name=self.name)

    def to_nan(self):
        return Point2DRelative(np.nan, np.nan, name=self.name)

    def is_nan(self):
        return np.isnan(self.x) or np.isnan(self.y)


def average_nan_same_name_points(*points):
    # print([point for point in points])
    if len(points) == 0:
        return None
    ave_point = Point2D(0., 0., name=points[0].name)
    # print([[_point.x, _point.y, _point.name] for _point in points])
    ave_point.x = np.nanmean([_point.x for _point in points])
    ave_point.y = np.nanmean([_point.y for _point in points])
    ave_point.conf = np.nanmean([_point.conf for _point in points])
    return ave_point


class LineSegment2D:
    def __init__(self, point_1, point_2):
        self.point_1 = point_1
        self.point_2 = point_2

    def length(self):
        return np.sqrt((self.point_1.x - self.point_2.x) ** 2 + (self.point_1.y - self.point_2.y) ** 2)

    def length_y(self):
        return np.abs((self.point_1.y - self.point_2.y))

    def mid_point(self, name=''):
        # print(np.shape(self.point_1.conf))
        return Point2D((self.point_1.x + self.point_2.x) / 2.0, (self.point_1.y + self.point_2.y) / 2.0,
                       conf=np.min([self.point_1.conf, self.point_2.conf], axis=0),
                       name=name)


def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


class ComplexEncoder(json.JSONEncoder):
    def default(self, obj):
        # print('complexEncoder')
        # print(type(obj))
        # print(hasattr(obj, 'class_dict'))
        if hasattr(obj, 'class_dict'):
            return obj.class_dict()
        else:
            return json.JSONEncoder.default(self, obj)


def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]


######################################################
# Time Lagged Cross Correlation
def crosscorr(datax, datay, lag=0, wrap=False):
    """ Lag-N cross correlation.
    Shifted data filled with NaNs

    Parameters
    ----------
    lag : int, default 0
    datax, datay : pandas.Series objects of equal length

    Returns
    ----------
    crosscorr : float
    """
    if wrap:
        shiftedy = datay.shift(lag)
        shiftedy.iloc[:lag] = datay.iloc[-lag:].values
        return datax.corr(shiftedy)
    else:
        return datax.corr(datay.shift(lag))


def calculate_MA(arr, window_size):
    # Program to calculate moving average
    # arr = [1, 2, 3, np.NaN, 9]
    # window_size = 3

    i = 0
    # Initialize an empty list to store moving averages
    moving_averages = []

    # Loop through the array to consider
    # every window of size 3
    while i < len(arr) - window_size + 1:
        # Store elements from i to i+window_size
        # in list to get the current window
        window = arr[i: i + window_size]

        # Calculate the average of current window
        window_average = np.nanmean(window)
        moving_averages.append(window_average)

        # Shift window to right by one position
        i += 1
    return moving_averages


def calculate_MA_array(arr, window_size):
    b = np.zeros(np.shape(arr))
    b[0:window_size, :] = arr[0:window_size, :]
    for i in range(np.shape(arr)[1]):
        b[window_size - 1:, i] = calculate_MA(arr[:, i], window_size)

    return b


# def update_MA(ma_series, new_value, window_size):
#     pass


# list of ranges of continuous NaN and Not NaN values
def continuous_nan_num_ranges(input_keypoint_series):
    nan_dict = {}
    num_dict = {}

    for key, data in input_keypoint_series.items():
        # print(key)
        # print(np.shape(data))
        if key not in ['__header__', '__version__', '__globals__']:
            row_old = np.zeros([3])
            i = 0
            nan_length = 0
            num_length = 0
            nan_begin = 0
            num_begin = 0
            nan_series_list = []
            num_series_list = []
            num_begin_value = 0

            for row in data:
                if np.isnan(row[0]):
                    if nan_length == 0:
                        nan_begin = i
                    if num_length > 0:
                        num_end_value = row_old[0:2]
                        num_series_dict = {'begin': num_begin, 'length': num_length,
                                           'start values': num_begin_value.tolist(),
                                           'end values': num_end_value.tolist()}
                        num_series_list.append(num_series_dict)
                        # print(num_series_list)
                        # input()
                        # if num_length <= 5:
                        #     print("ATTENTION: num length", num_length, "starting", num_begin, "at", key)
                        num_length = 0
                    nan_length += 1
                else:
                    if num_length == 0:
                        num_begin = i
                        num_begin_value = row[0:2]
                    if nan_length > 0:
                        nan_series_list.append((nan_begin, nan_length))
                        nan_length = 0
                    num_length += 1

                i += 1
                row_old = row
            if num_length > 0:
                num_end_value = row_old[0:2]
                num_series_dict = {'begin': num_begin, 'length': num_length,
                                   'start values': num_begin_value.tolist(),
                                   'end values': num_end_value.tolist()}
                num_series_list.append(num_series_dict)
                # if num_length <= 5:
                #     print("ATTENTION: num length", num_length, "starting", num_begin, "at", key)
                # num_length = 0
            if nan_length > 0:
                nan_series_list.append((nan_begin, nan_length))
                # nan_length = 0
            nan_dict[key] = nan_series_list
            num_dict[key] = num_series_list

    return nan_dict, num_dict


def filter_sub_series(full_series, window_length=5):
    nan_dict, num_dict = continuous_nan_num_ranges({'ex': full_series})
    for _sub in num_dict['ex']:
        _sub_series = full_series[_sub['begin']:_sub['begin'] + _sub['length']]
        _sub_series_filtered = savgol_filter(_sub_series, window_length=window_length,
                                             polyorder=2, mode='nearest', axis=0)
        full_series[_sub['begin']:_sub['begin'] + _sub['length']] = _sub_series_filtered
        # print(data_.shape)
    return full_series


def interpolate_sub_series(sub_series, method="linear", order=None):
    time_series = pd.DataFrame(sub_series)
    time_series_interp = time_series.interpolate(method=method, order=order, axis=0).to_numpy()
    return time_series_interp


def short_nan_sub_series(full_series, maximum_nan_length=20):
    nan_dict, num_dict = continuous_nan_num_ranges({'ex': full_series})
    list_of_sub_series_contain_short_nan = []
    first_ = [0, 0]
    second_ = [0, 0]
    # print(nan_dict)
    for _range in nan_dict['ex']:

        if _range[1] > maximum_nan_length:
            # print(_range)
            first_[1] = _range[0]
            second_[0] = _range[0] + _range[1]
            # print('first is', first_)
            # print('second is', second_)
            if first_[0] != first_[1]:
                list_of_sub_series_contain_short_nan.append(first_)
            first_ = second_[:]
            # print('first after', first_)
    second_[1] = full_series.shape[0]
    if second_[0] != second_[1]:
        list_of_sub_series_contain_short_nan.append(second_)
    # for _sub in list_of_sub_series_contain_short_nan:
    #     if _sub[0] == _sub[1]:
    #         list_of_sub_series_contain_short_nan.remove(_sub)
    # print(list_of_sub_series_contain_short_nan)
    return list_of_sub_series_contain_short_nan


def normalize_column(arr):
    _, no_cols = np.shape(arr)
    res = np.zeros(np.shape(arr))
    # iterate over columns using Fortran order
    for i, col in enumerate(arr.T):
        # print(col)
        max_col = np.nanmax(col)
        min_col = np.nanmin(col)
        # res = copy.deepcopy(col)
        if min_col < 0:
            # res.applymap(lambda x: x / (2 * max_col) + 0.5 if x >= 0 else x / (2 * np.abs(min_col)) + 0.5)
            col[col >= 0] = col[col >= 0] / (2 * max_col) + 0.5
            col[col < 0] = col[col < 0] / (2 * np.abs(min_col)) + 0.5
        else:
            col = (col - min_col) / (max_col - min_col)
        res[:, i] = col
    # res = np.nan_to_num(res, nan=-1.0)
    return res


def point_assignment(a, b):
    from scipy.spatial.distance import cdist

    # assuming 'a' and 'b' are lists of 2D points, where each point is represented as a tuple of (x, y) coordinates

    # convert the lists of points to NumPy arrays
    a = np.array(a)
    b = np.array(b)

    # calculate the distances between all pairs of points using cdist
    distances = cdist(a, b)

    # find the indices of the closest point in list 'b' for each point in list 'a'
    closest_indices = np.argmin(distances, axis=1)

    # create a dictionary that maps each point in list 'a' to the closest point in list 'b'
    mapping = {}
    for i, idx in enumerate(closest_indices):
        if idx not in mapping.values():
            mapping[tuple(a[i])] = idx

    # find the unassigned points in list 'a'
    unassigned_a = [tuple(a[i]) for i in range(len(a)) if tuple(a[i]) not in mapping.keys()]

    # find the unassigned points in list 'b'
    unassigned_b_i = [i for i in range(len(b)) if i not in mapping.values()]

    # if there are unassigned points in list 'a' and unassigned points in list 'b', assign them to each other
    if len(unassigned_a) > 0 and len(unassigned_b_i) > 0:
        # calculate the distances between all pairs of unassigned points using cdist
        distances = cdist(unassigned_a, b[unassigned_b_i])

        # find the indices of the closest unassigned point in list 'b' for each unassigned point in list 'a'
        closest_indices = np.argmin(distances, axis=1)

        # assign the unassigned points to each other
        for i, idx in enumerate(closest_indices):
            mapping[unassigned_a[i]] = idx

    # print the mapping
    print(mapping)


def assign_points(a, b):
    if not a or not b:
        return {}
    dist_matrix = np.linalg.norm(np.nan_to_num(np.array(a)[:, np.newaxis] - np.array(b), nan=sys.maxsize), axis=2)
    row_ind, col_ind = linear_sum_assignment(dist_matrix)
    mapping = {i: j for i, j in zip(row_ind, col_ind)}
    return mapping
