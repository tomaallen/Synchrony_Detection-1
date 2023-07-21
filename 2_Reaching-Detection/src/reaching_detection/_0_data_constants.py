import os

const_dir_path = os.path.dirname(os.path.realpath(__file__))
# print(dir_path)

INPUT_FOLDER = const_dir_path + '\\..\\..\\data\\list_of_files\\'  # 'data/lp007/'  # Note: slash (/) should be at
# the end #'demo12_1_2/'  #
OUTPUT_FOLDER = const_dir_path + '\\..\\..\\data\\output_files\\'

ORIGINAL_VIDEO_NAME = 'Camcorder 2 DEmo.mp4_skeleton.avi' # 'Camcorder 1_Eval room_STT.avi'  #  Note: include the
# file extension #
PREFIX = ORIGINAL_VIDEO_NAME[:ORIGINAL_VIDEO_NAME.rindex('.')] + '_'
JSONL_FILE = 'output.json'
JSON_FILE = 'PD-output.json'

MATLAB_RAW_DATA_FILE = 'raw_data_keypoints.mat'
MATLAB_PREPROCESSED_DATA_FILE = 'preprocessed_data_keypoints.mat'
MATLAB_PREPROCESSED_DATA_FILE_REMOVE_SHORT_NONAN = 'preprocessed_data_keypoints_remove_short_nonan' \
                                                   '.mat '

JSON_NAN_LIST = 'preprocessed_data_keypoints_nan_list.jsonl'
JSON_NONAN_LIST = 'preprocessed_data_keypoints_nonan_list.jsonl'

MATLAB_VECTORS = 'vector_matrix.mat'
MATLAB_ANGLES = 'angle_matrix.mat'

MATLAB_TRAINING_INPUT_NO_NAN = "reaching_training_input_no_nan.mat"
MATLAB_TRAINING_OUTPUT_NO_NAN = 'reaching_training_labels_no_nan.mat'
MATLAB_TEST_INPUT_NO_NAN = 'reaching_test_input_no_nan.mat'
MATLAB_TEST_OUTPUT_NO_NAN = 'reaching_test_labels_no_nan.mat'

MATLAB_TRAINING_INPUT_NAN = "reaching_training_input_nan.mat"
MATLAB_TRAINING_OUTPUT_NAN = 'reaching_training_labels_nan.mat'
MATLAB_TEST_INPUT_NAN = 'reaching_test_input_nan.mat'
MATLAB_TEST_INPUT_NAN_ADD_NAN = 'reaching_test_input_nan_add_nan_at_col13.mat'
MATLAB_TEST_OUTPUT_NAN = 'reaching_test_labels_nan.mat'

MATLAB_TRAINING_INPUT_NAN_MONO = "reaching_training_input_nan_mono.mat"
MATLAB_TRAINING_OUTPUT_NAN_MONO = 'reaching_training_labels_nan_mono.mat'
MATLAB_TEST_INPUT_NAN_MONO = 'reaching_test_input_nan_mono.mat'
MATLAB_TEST_OUTPUT_NAN_MONO = 'reaching_test_labels_nan_mono.mat'

MATLAB_TRAINING_INPUT_NAN_STEREO = "reaching_training_input_nan_stereo.mat"
MATLAB_TRAINING_OUTPUT_NAN_STEREO = 'reaching_training_labels_nan_stereo.mat'
MATLAB_TEST_INPUT_NAN_STEREO = 'reaching_test_input_nan_stereo.mat'
MATLAB_TEST_OUTPUT_NAN_STEREO = 'reaching_test_labels_nan_stereo.mat'

MATLAB_TRAINING_INPUT_ALL_NUMBER_MONO = "reaching_training_input_all_number_mono.mat"
MATLAB_TRAINING_OUTPUT_ALL_NUMBER_MONO = 'reaching_training_labels_all_number_mono.mat'
MATLAB_TEST_INPUT_ALL_NUMBER_MONO = 'reaching_test_input_all_number_mono.mat'
MATLAB_TEST_OUTPUT_ALL_NUMBER_MONO = 'reaching_test_labels_all_number_mono.mat'

MATLAB_TRAINING_INPUT_ALL_NUMBER_STEREO = "reaching_training_input_all_number_stereo.mat"
MATLAB_TRAINING_OUTPUT_ALL_NUMBER_STEREO = 'reaching_training_labels_all_number_stereo.mat'
MATLAB_TEST_INPUT_ALL_NUMBER_STEREO = 'reaching_test_input_all_number_stereo.mat'
MATLAB_TEST_OUTPUT_ALL_NUMBER_STEREO = 'reaching_test_labels_all_number_stereo.mat'

XGB_MODEL = 'xgb_reaching.json'
XGB_MODEL_NAN = 'xgb_reaching_nan.json'

KEYPOINTS_DICT = {0: "Nose",
                  1: "Neck",
                  2: "RShoulder",
                  3: "RElbow",
                  4: "RWrist",
                  5: "LShoulder",
                  6: "LElbow",
                  7: "LWrist",
                  8: "MidHip",
                  9: "RHip",
                  10: "RKnee",
                  11: "RAnkle",
                  12: "LHip",
                  13: "LKnee",
                  14: "LAnkle",
                  15: "REye",
                  16: "LEye",
                  17: "REar",
                  18: "LEar",
                  19: "LBigToe",
                  20: "LSmallToe",
                  21: "LHeel",
                  22: "RBigToe",
                  23: "RSmallToe",
                  24: "RHeel"}

CSV_HEADER_LST = []
for id, data in KEYPOINTS_DICT.items():
    CSV_HEADER_LST.append(data+': X')
    CSV_HEADER_LST.append( data + ': Y')
    CSV_HEADER_LST.append(data + ': conf')
CSV_HEADER_LST.insert(0, 'IDs')
CSV_HEADER = ','.join(CSV_HEADER_LST)

TRACK_ID = False
SKELETON = False
KEYPOINT_ONLY = False
NO_HEADER = False
OUTPUT_TYPE = '.mp4'


# CSV_FOLDER = OUTPUT_FOLDER + 'csv_files\\'
#
# if not os.path.exists(CSV_FOLDER):
#     os.makedirs(CSV_FOLDER)
#
# NO_CSV_FILES = len([name for name in os.listdir(CSV_FOLDER)
#                     if os.path.isfile(os.path.join(CSV_FOLDER, name))]) // 2

