import copy
import json
import numpy as np
import os
import constants
import utils
from utils import ComplexEncoder, assign_points
from human_properties import Person, average_multi_person_ma, body_base_distance
import queue
import pandas as pd
from scipy.io import loadmat, savemat
from utils import file_path_giving_folder
import argparse


def read_csv(filename=None):
    # Read the CSV file
    if filename is None:
        filename = file_path_giving_folder("csv_face", parent_dir=input_folder)[0]
    df = pd.read_csv(filename)
    # df = df.drop(0)
    # Display the data
    print(df.head())

    # Convert the pandas DataFrame to a dictionary
    data_array = df.to_numpy()
    print(data_array.shape)
    data_array[data_array == -1] = np.nan
    data_mom = data_array[:, 0:2]
    data_baby = data_array[:, 2:4]
    data_mom = np.c_[data_mom, np.ones((data_mom.shape[0], 1)) * 0.5]
    data_baby = np.c_[data_baby, np.ones((data_baby.shape[0], 1)) * 0.5]
    # data_mom_dict = {'Neck': data_mom}
    # data_baby_dict = {'Neck': data_baby}
    data_all = {'0': data_mom, '1': data_baby}

    for _id, _data in data_all.items():
        # Save the data to a MATLAB .mat file
        if not os.path.exists(os.path.join(input_folder, "matlab_files", str(_id))):
            os.makedirs(os.path.join(input_folder, "matlab_files", str(_id)))

        savemat(os.path.join(input_folder, "matlab_files", str(_id),
                             os.path.basename(input_folder) + '-' + constants.MATLAB_RAW_DATA_FILE), {'Face': _data})
    return df


def compute_score_face(frame, list_of_person_faces, face_data_yolo):
    # if 5165 < int(frame) < 5185:
    #     print(list_of_person_faces, face_data_yolo)

    mapping = assign_points(list_of_person_faces, face_data_yolo)
    # if 5165 < int(frame) < 5185:
    #     print(mapping)
    #     input()

    closeness_to_baby_face_score = [-1] * len(list_of_person_faces)

    for _id, _value in mapping.items():
        # print(type(int(_id)), _id)
        # print(type(_value), _value)
        closeness_to_baby_face_score[int(_id)] = int(_value)  # convert from numpy.int64 to int
    # print(closeness_to_baby_face_score)
    # input()
    return closeness_to_baby_face_score


def get_score_for_persons_based_on_face(list_of_frames, json_data, json_score_only, face_data):
    scores_frames = {}
    for _frame, _data in json_data.items():
        # print('frame:', _frame)
        if int(_frame) in list_of_frames:
            # print('====================================================')
            print('frame number:', _frame)
            # if 3800 < int(_frame) < 3820:
            #     print('===========================================================================')
            #     print(_frame)
            list_of_person = []
            list_of_person_faces = []
            for _person_id, _pose in _data['Data'].items():
                person_temp = Person()
                person_temp.input_skeleton(_person_id, _pose)
                list_of_person.append(person_temp)
                list_of_person_faces.append(person_temp.face_bounding_box_center())

            current_faces = [face_data[i][int(_frame) - 1, 0:2] for i in [0, 1]]
            # the row int(_frame)-1, because of numpy array index is less than frame index by 1

            face_closeness_scores \
                = compute_score_face(_frame,
                                     list_of_person_faces,
                                     current_faces)

            for i, _person in enumerate(list_of_person):
                # pass
                _person.baby_MA_ID = face_closeness_scores[i]
                # print(type(_person.baby_MA_ID))
                # input()

            # print(scores[i])
            if json_score_only == 1:
                new_dict_persons = {}
                for k, _person in enumerate(list_of_person):
                    new_dict_persons[_person.ID] = [_person.closeness_score, _person.head_ratio_score,
                                                    _person.total_score, _person.baby_MA_ID]
                subdict_of_frame = {"Count": _data["Count"], "Data": new_dict_persons}
                scores_frames[_frame] = subdict_of_frame
            else:
                new_dict_persons = []
                for k, _person in enumerate(list_of_person):
                    __person = copy.deepcopy(_person)
                    __person.Body.body_zero()
                    # print(type(__person))
                    # print(json.dumps(__person, cls=ComplexEncoder))
                    # input()
                    new_dict_persons.append(json.loads(json.dumps(__person, cls=ComplexEncoder)))
                    # if 30 < int(_frame) < 76:
                    #     # print('=======================================')
                    #     # print(_frame)
                    #     print(json.dumps(__person, cls=ComplexEncoder))
                    #     print("close", __person.closeness_score,
                    #           "ratio", __person.head_ratio_score,
                    #           "total", __person.total_score,
                    #           "ma_id", __person.baby_MA_ID)
                subdict_of_frame = {"Count": _data["Count"], "Data": new_dict_persons}
                scores_frames[_frame] = subdict_of_frame
            # input()
    return scores_frames


def poses_for_frames(list_of_frames=None, output_json_folder=None, json_score_only=1):
    # arguments:
    # list_of_frames: Eg. range(10, 20)
    # list all file paths in folder "json_files"

    json_file = utils.file_path_giving_folder("json_files", parent_dir=input_folder)
    # load the json file into a dictionary
    for _file in json_file:
        if constants.JSON_FILE in _file:
            print(_file)
            with open(_file, 'r') as f:
                json_data = json.load(f)

    no_csv_files = len(json_data)
    # print(no_csv_files)
    if list_of_frames is None:
        list_of_frames = range(0 + 1, no_csv_files + 1)

    print(list_of_frames)
    face_dict = read_matlab_yolo(data_type=constants.MATLAB_RAW_DATA_FILE)
    face_data = []
    for _id in ['0', '1']:
        face_data.append(face_dict[_id])

    scores_frames = get_score_for_persons_based_on_face(list_of_frames, json_data, json_score_only, face_data)

    if json_score_only == 1:
        suf_name = constants.JSON_SCORE_FILE
    elif json_score_only == 2:
        suf_name = constants.JSON_SCORE_FILE_2
    else:
        suf_name = constants.JSON_COMBINED_FILE

    if output_json_folder is None:
        output_json_folder = os.path.join(input_folder, "json_files")
    # print('output_json_folder', output_json_folder)
    with open(os.path.join(output_json_folder, os.path.basename(input_folder) + '-' + suf_name), 'w') as fp:
        # print(scores_frames)
        json.dump(scores_frames, fp)

    # print(data)


def read_matlab_yolo(data_type=constants.MATLAB_RAW_DATA_FILE, save_face=False):
    frame_width, frame_height, frame_fps, total_no_frames = utils.read_video_info(input_folder)
    print(frame_width, frame_height)
    face_data = {}
    for _id in ['0', '1']:
        # Load the face data from the face folder
        arr = loadmat(os.path.join(input_folder, "matlab_files", str(_id),
                                   os.path.basename(input_folder) + '-' + data_type))['Face']
        arr[:, 0] *= frame_width  # multiply the first column by 'a'
        arr[:, 1] *= frame_height  # multiply the second column by 'b
        face_data[_id] = arr
        if save_face:
            savemat(os.path.join(input_folder, "matlab_files", str(_id),
                                 os.path.basename(input_folder) + '-' + constants.MATLAB_FACE_TO_OPENPOSE), {'Face': arr})
    # print(face_data)
    # input()

    return face_data


if __name__ == "__main__":

    my_parser = argparse.ArgumentParser(description='Process some arguments')
    # Add the arguments
    my_parser.add_argument('--reach_dir',
                           type=str,
                           help='video specific reaching detection output folder with csv face',
                           default=r'E:\AnB_Pose\pose_detect_output\BLBR0001_SyncCam_trial1')

    # Execute the parse_args() method
    args = my_parser.parse_args()

    input_folder = args.reach_dir
    read_csv()
    poses_for_frames(json_score_only=0)
    print(os.path.join(args.reach_dir, 'json_files', os.path.basename(args.reach_dir) + '-PD-combined_output.json'))
    print('combining json for ' + os.path.basename(args.reach_dir))
