# -*- coding: utf-8 -*-
import imp
from subprocess import check_output

import cv2
import sys

import time
import numpy as np
import os

# print('in the input: ', sys.path)
dir_path = os.path.dirname(os.path.realpath(__file__))
# print(dir_path)
try:
    # Change these variables to point to the correct folder (Release/x64 etc.)
    sys.path.append(dir_path + '/../bin/python/openpose/Release')
    ex_path = os.environ['PATH'] + ';' + dir_path + '/../bin/python/openpose/Release;' + dir_path + '/../bin;'
    # print(ex_path)
    os.environ['PATH'] = ex_path
    import pyopenpose as op

except ImportError as e:
    print(
        'Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    raise e
import pygame
# Load OpenPose:
# sys.path.append('/usr/local/python')
# from openpose import pyopenpose as op
# Import Openpose (Windows/Ubuntu/OSX)


from deep_sort.iou_matching import iou_cost
from deep_sort.kalman_filter import KalmanFilter
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker as DeepTracker
from deep_sort import nn_matching
from deep_sort import preprocessing
from deep_sort.linear_assignment import min_cost_matching
from deep_sort.detection import Detection as ddet
from tools import generate_detections as gdet
from utils import poses2boxes, isMediaFile, get_iou

import Constants
import reaching_detection._0_data_constants as reaching_const

body_25_colors = [(255, 0, 85), (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0), (85, 255, 0),
                  (0, 255, 0), (255, 0, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
                  (0, 0, 255), (255, 0, 170), (170, 0, 255), (255, 0, 255), (85, 0, 255), (0, 0, 255), (0, 0, 255),
                  (0, 0, 255), (0, 255, 255), (0, 255, 255), (0, 255, 255)]

video_name = 'Camcorder 2 DEmo.mp4'
video_name_no_extension = video_name[:video_name.rindex('.')]
# output_video_extension = '.avi'  # .avi, .mp4
output_image_extension = '.jpg'


# input_video_path = os.path.join(reaching_const.INPUT_FOLDER, video_name)
# output_video_path = os.path.join(reaching_const.INPUT_FOLDER, video_name_no_extension + suffix + output_video_extension)

# total_preds = np.load('total_pred.npy')
def keypoint_filter(keypoint_array):
    list_of_out = []
    for i in range(keypoint_array.shape[0]):
        if any(keypoint_array[i, 1, :] != 0) and ((any(keypoint_array[i, 2, :] != 0) and any(
                keypoint_array[i, 3, :] != 0) and any(keypoint_array[i, 4, :] != 0)) or (
                                                          any(keypoint_array[i, 5, :] != 0) and any(
                                                          keypoint_array[i, 6, :] != 0) and any(
                                                          keypoint_array[i, 7, :] != 0))):
            pass
        else:
            list_of_out.append(i)
    return np.delete(keypoint_array, list_of_out, axis=0)

def keypoint_filter_full_body(keypoint_array):
    list_of_out = []
    for i in range(keypoint_array.shape[0]):
        if keypoint_array[i].shape[0]>=7:
            pass
        else:
            list_of_out.append(i)
    return np.delete(keypoint_array, list_of_out, axis=0)


class Input():
    def __init__(self, debug=False, file_name=None):
        if reaching_const.SKELETON:
            suffix = '-skeleton'
        elif reaching_const.KEYPOINT_ONLY:
            suffix = '-keypoints'
        else:
            suffix = '-output'
        # from openpose import *
        # print('-----START', reaching_const.INPUT_FOLDER, file_name)
        self.input_is_image = False
        self.input_is_video = False
        if isMediaFile(file_name) == 'video':
            self.input_is_video = True
        elif isMediaFile(file_name) == 'image':
            self.input_is_image = True
        else:
            print(isMediaFile(file_name))
        self.video_input = os.path.join(reaching_const.INPUT_FOLDER, file_name)
        prefix = file_name[:file_name.rindex('.')]
        folder_csv = reaching_const.OUTPUT_FOLDER + prefix + '/'

        self.folder_csv = folder_csv + 'csv_files/'
        if not os.path.exists(self.folder_csv):
            os.makedirs(self.folder_csv)

        self.folder_output = folder_csv + 'output_videos/'
        if not os.path.exists(self.folder_output):
            os.makedirs(self.folder_output)

        if not os.path.exists(folder_csv + 'output_videos_skeleton/'):
             os.makedirs(folder_csv + 'output_videos_skeleton/')
             
        self.video_info = folder_csv + 'video_info/'
        if not os.path.exists(self.video_info):
            os.makedirs(self.video_info)
        if self.input_is_image:
            self.video_output = folder_csv + 'output_videos/' + prefix + suffix + output_image_extension
        else:
            if reaching_const.SKELETON:
                self.video_output = folder_csv + 'output_videos_skeleton/' + prefix + suffix + reaching_const.OUTPUT_TYPE
            else:
                self.video_output = folder_csv + 'output_videos/' + prefix + suffix + reaching_const.OUTPUT_TYPE
        params = dict()
        params["model_folder"] = Constants.openpose_modelfolder
        params["net_resolution"] = "-1x432"
        # params["hand"] = True
        if reaching_const.SKELETON or reaching_const.KEYPOINT_ONLY:
            params["disable_blending"] = True
        if reaching_const.KEYPOINT_ONLY:
            params["alpha_pose"] = 0

        self.openpose = op.WrapperPython()
        self.openpose.configure(params)
        self.openpose.start()

        max_cosine_distance = Constants.max_cosine_distance
        nn_budget = Constants.nn_budget
        self.nms_max_overlap = Constants.nms_max_overlap
        max_age = Constants.max_age
        n_init = Constants.n_init
        model_filename = dir_path + '/model_data/mars-small128.pb'
        self.encoder = gdet.create_box_encoder(model_filename, batch_size=1)
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = DeepTracker(metric, max_age=max_age, n_init=n_init)

        if self.input_is_video:
            self.capture = cv2.VideoCapture(self.video_input)
            # print(self.video_file)
            self.frame_width = int(self.capture.get(3))
            self.frame_height = int(self.capture.get(4))
            frame_rate = (self.capture.get(5))

            if self.capture.isOpened():  # Checks the stream
                self.frameSize = (int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                                  int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
            Constants.SCREEN_HEIGHT = self.frameSize[0]
            Constants.SCREEN_WIDTH = self.frameSize[1]
        else:
            # read image
            self.capture = cv2.imread(self.video_input, cv2.IMREAD_UNCHANGED)

            # get dimensions of image
            dimensions = self.capture.shape

            # height, width, number of channels in image
            self.frame_height = self.capture.shape[0]
            self.frame_width = self.capture.shape[1]
            channels = self.capture.shape[2]
            Constants.SCREEN_HEIGHT = self.frame_height
            Constants.SCREEN_WIDTH = self.frame_width


        if self.frame_height > 720:
            self.size = (int(self.frame_width*720/self.frame_height), 720)
        else:
            self.size = (self.frame_width, self.frame_height)

        if self.input_is_video:
            lines = ['Width: ' + str(self.frame_width), 'Height: ' + str(self.frame_height), 'FPS: ' + str(frame_rate) ,'Size of output: ' + str(self.size)]
        else:
            lines = ['Width: ' + str(self.frame_width), 'Height: ' + str(self.frame_height), 'dimensions: ' + str(dimensions) ,'Size of output: ' + str(self.size)]
        with open(os.path.join(self.video_info, 'readme.txt'), 'w') as f:
            for line in lines:
                f.write(line)
                f.write('\n')

        
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') if reaching_const.OUTPUT_TYPE == '.mp4' else cv2.VideoWriter_fourcc(*'MJPG')
        if self.input_is_video:
            self.result_video = cv2.VideoWriter(self.video_output,
                                                fourcc,
                                                frame_rate, self.size)

    def getCurrentFrameAsImage(self):
        frame = self.currentFrame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pgImg = pygame.image.frombuffer(frame.tostring(), frame.shape[1::-1], "RGB")
        return pgImg

    def run(self, no_frames):
        if self.input_is_video:
            result, self.currentFrame = self.capture.read()
        else:
            self.currentFrame = self.capture
            if no_frames == 0:
                result = 1
            else:
                result = 0

        if not result:
            print("Can't receive frame (stream end?). Exiting ...")
            print("==========================================================================")
            lines = ['Total no. of frames: '+ str(no_frames)]
            with open(os.path.join(self.video_info, 'readme.txt'), 'a') as f:
                for line in lines:
                    f.write(line)
                    f.write('\n')
            return 0
        datum = op.Datum()
        datum.cvInputData = self.currentFrame
        self.openpose.emplaceAndPop(op.VectorDatum([datum]))

        keypoints, self.currentFrame = np.array(datum.poseKeypoints), datum.cvOutputData
        # print(np.shape(keypoints))
        # input()
        # Doesn't use keypoint confidence
        try:
            keypoints = keypoint_filter(keypoints)
            poses = keypoints[:, :, :2]
            # Get containing box for each seen body
            boxes = poses2boxes(poses)
        except Exception as e:
            print(e)
            poses = np.zeros([1, 25, 3])
            boxes = []
        # print('No of poses:',len(boxes))
        boxes_xywh = [[x1, y1, x2 - x1, y2 - y1] for [x1, y1, x2, y2] in boxes]
        features = self.encoder(self.currentFrame, boxes_xywh)
        # print(features)

        nonempty = lambda xywh: xywh[2] != 0 and xywh[3] != 0
        detections = [Detection(bbox, 1.0, feature, pose) for bbox, feature, pose in zip(boxes_xywh, features, poses) if
                      nonempty(bbox)]
        # Run non-maxima suppression.
        boxes_det = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes_det, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        # Call the tracker
        self.tracker.predict()
        self.tracker.update(detections)

        track_id = []
        track_id_sort = []
        sort_boxes = {}

        # print(boxes)
        boxes_dict = [{'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2} for [x1, y1, x2, y2] in boxes]
        # print(boxes_dict)
        # print(sort_boxes)
        max_ious = [0] * len(boxes_dict)
        max_ious_id = [0] * len(boxes_dict)
        max_iou_boxes = [0] * len(boxes_dict)
        colors = [0] * len(boxes_dict)
        # print(max_ious)
        # input()

        for track in self.tracker.tracks:
            color = None
            if not track.is_confirmed():
                color = (0, 0, 255)
            else:
                color = (255, 255, 255)
            bbox = track.to_tlbr()

            sort_box = {'x1': int(bbox[0]), 'y1': int(bbox[1]), 'x2': int(bbox[2]), 'y2': int(bbox[3])}
            for i, bb_ in enumerate(boxes_dict):
                iou_bb_ = get_iou(bb_, sort_box)
                if iou_bb_ > max_ious[i]:
                    max_ious[i] = iou_bb_
                    max_ious_id[i] = track.track_id
                    max_iou_boxes[i] = sort_box
                    colors[i] = color
            # print(max_ious, max_ious_id)
            # input()
            # print("Body keypoints:")
            # print(track.last_seen_detection.pose)
            # cv2.rectangle(self.currentFrame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),color, 2)
            # if no_frames == 450:
            #     print((int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])))
            # cv2.putText(self.currentFrame, "%s"%(track.track_id),(int(bbox[0]), int(bbox[1])-5),0, 5e-3 * 200, (0,255,0),2) # - ts%s # ,track.time_since_update
            track_id_sort.append(track.track_id)

            sort_boxes[track.track_id] = {'x1': int(bbox[0]), 'y1': int(bbox[1]), 'x2': int(bbox[2]),
                                          'y2': int(bbox[3])}

        # # print(max_iou_boxes, max_ious_id,colors)
        # print deepsort rectangles

        if reaching_const.SKELETON or reaching_const.KEYPOINT_ONLY:
            # data_id = np.genfromtxt(self.folder_csv + '_track' + str(no_frames + 1) + '.csv', delimiter=',')

            data_id = np.genfromtxt(self.folder_csv + "point" + str(no_frames + 1) + ".csv",
                                    delimiter=",")
            while data_id.ndim < 2:
                data_id = np.expand_dims(data_id, axis=0)
            # print('+++++++++++++++++++++++++++',data_id[:,0], data_id.shape)
            # input()
            try:
                data_id = data_id[:, 0]
                for i in range(len(boxes_dict)):
                    # print(i)
                    # print(max_iou_boxes[i],colors[i])
                    cv2.rectangle(self.currentFrame, (int(max_iou_boxes[i]['x1']), int(max_iou_boxes[i]['y1'])),
                                  (int(max_iou_boxes[i]['x2']), int(max_iou_boxes[i]['y2'])), colors[i], 2)
                    cv2.putText(self.currentFrame, "%s" % (int(data_id[i])),
                                (int(max_iou_boxes[i]['x1']), int(max_iou_boxes[i]['y1']) - 5), 0, 5e-3 * 200*self.frame_height/720,
                                (255, 255, 255), 2)
            except Exception as e:
                print(e)

        if not (reaching_const.SKELETON or reaching_const.KEYPOINT_ONLY):
            try:
                for i in range(len(boxes_dict)):
                    # print(i)
                    # print(max_iou_boxes[i],colors[i])
                    cv2.rectangle(self.currentFrame, (int(max_iou_boxes[i]['x1']), int(max_iou_boxes[i]['y1'])),
                                  (int(max_iou_boxes[i]['x2']), int(max_iou_boxes[i]['y2'])), colors[i], 2)
                    cv2.putText(self.currentFrame, "%s" % (max_ious_id[i]),
                                (int(max_iou_boxes[i]['x1']), int(max_iou_boxes[i]['y1']) - 5), 0, 5e-3 * 200*self.frame_height/720,
                                (255, 0, 0), 2)
            except Exception as e:
                print(e)

        cv2.putText(self.currentFrame, "%s" % (no_frames + 1), (int(5*self.frame_height/720), int(25*self.frame_height/720)), 0, 5e-3 * 200*self.frame_height/720, (255, 0, 0), 2)

        # if total_preds[no_frames] >= 0.4:
        #     pred_color = (255,255,255)
        # else:
        #     pred_color = (255,0,0)
        # cv2.putText(self.currentFrame, "%.2g"%(total_preds[no_frames]), (self.size[0] - self.size[0]//2 - 30, self.size[1]-30),0, 5e-3 * 200, pred_color,2)
        # print(poses)

        ## keypoints only
        if reaching_const.KEYPOINT_ONLY:
            try:
                for i, xy in enumerate(np.reshape(poses, [-1, 2])):
                    if int(xy[0]) > 0 or int(xy[1]) > 0:
                        # print(x,y)
                        # print(type(x))
                        cv2.circle(self.currentFrame, (int(xy[0]), int(xy[1])), radius=5, color=body_25_colors[i % 25],
                                   thickness=-1)
            except Exception as e:
                print(e)

        # print('new',max_ious, max_ious_id)

        for bb_ in boxes_dict:
            try:
                max_ = max(sort_boxes, key=lambda k: get_iou(bb_, sort_boxes[k]))
                # print(max_)
                track_id.append(max_)
            except Exception as e:
                # print('=====================This is the frame==================', no_frames)
                print(e)
                # print(sort_boxes)
                input()
        # input()

        arr_track_id = np.array(track_id)
        # print('Track ids', arr_track_id, arr_track_id.shape)
        # print('Track ids of sort', track_id_sort)
        if arr_track_id.shape[0] == 0:
            arr_track_id = np.array([-1])
        # print('Track ids', arr_track_id, arr_track_id.shape)

        if not (reaching_const.SKELETON or reaching_const.KEYPOINT_ONLY):
            if reaching_const.TRACK_ID:
                np.savetxt(self.folder_csv + "_track" + str(no_frames + 1) + ".csv", arr_track_id, fmt='%.6g',
                           delimiter=",")
            try:
                _header = '' if reaching_const.NO_HEADER else reaching_const.CSV_HEADER
                data_to_save = np.reshape(keypoints, [np.shape(keypoints)[0], -1])
                data_to_save = np.c_[arr_track_id, data_to_save]
                # print('cs point', data_to_save.shape)
                np.savetxt(self.folder_csv + "point" + str(no_frames + 1) + ".csv", data_to_save, fmt='%.6g',
                           delimiter=",",
                           header=_header, comments='')
            except Exception as e:
                print(e)
                _header = '' if reaching_const.NO_HEADER else reaching_const.CSV_HEADER
                data_to_save = np.zeros([1, 75])
                data_to_save = np.c_[arr_track_id, data_to_save]
                # print('cs point', data_to_save.shape)
                np.savetxt(self.folder_csv + "point" + str(no_frames + 1) + ".csv", data_to_save, fmt='%.6g',
                           delimiter=",",
                           header=_header, comments='')
        # if no_frames == 450:
        #     print(boxes_xywh)
        #     input()
        if self.input_is_video:
            resize_frame = cv2.resize(self.currentFrame, self.size)
            self.result_video.write(resize_frame)
        elif self.input_is_image:
            # Filename
            # filename = 'savedImage.jpg'

            # Using cv2.imwrite() method
            # Saving the image
            cv2.imwrite(self.video_output, self.currentFrame)

        if cv2.waitKey(1) == ord('p'):
            input()
