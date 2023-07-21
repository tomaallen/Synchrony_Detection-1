# -*- coding: utf-8 -*-
import cv2
import sys
import time
import numpy as np
import os

import pygame
# Load OpenPose:
# sys.path.append('/usr/local/python')
# from openpose import pyopenpose as op
# Import Openpose (Windows/Ubuntu/OSX)
dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)
try:
    # Change these variables to point to the correct folder (Release/x64 etc.)
    sys.path.append(dir_path + '/../bin/python/openpose/Release')
    ex_path  = os.environ['PATH'] + ';' + dir_path + '/../bin/python/openpose/Release;' +  dir_path + '/../bin;'
    print(ex_path)
    os.environ['PATH'] = ex_path
    import pyopenpose as op
except ImportError as e:
    print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    raise e


from deep_sort.iou_matching import iou_cost
from deep_sort.kalman_filter import KalmanFilter
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker as DeepTracker
from deep_sort import nn_matching
from deep_sort import preprocessing
from deep_sort.linear_assignment import min_cost_matching
from deep_sort.detection import Detection as ddet
from tools import generate_detections as gdet
from utils import poses2boxes

import Constants

def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    if bb1['x1'] > bb1['x2']:
        bb1['x1'] , bb1['x2'] = bb1['x2'] , bb1['x1']
    if bb1['y1'] > bb1['y2']:
        bb1['y1'] , bb1['y2'] = bb1['y2'] , bb1['y1']
    if bb2['x1'] > bb2['x2']:
        bb2['x1'] , bb2['x2'] = bb2['x2'] , bb2['x1']
    if bb2['y1'] > bb2['y2']:
        bb2['y1'] , bb2['y2'] = bb2['y2'] , bb2['y1']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

class Input():
    def __init__(self, debug = False):
        #from openpose import *
        params = dict()
        params["model_folder"] = Constants.openpose_modelfolder
        params["net_resolution"] = "-1x320"
        params["disable_blending"] = True
        self.openpose = op.WrapperPython()
        self.openpose.configure(params)
        self.openpose.start()


        max_cosine_distance = Constants.max_cosine_distance
        nn_budget = Constants.nn_budget
        self.nms_max_overlap = Constants.nms_max_overlap
        max_age = Constants.max_age
        n_init = Constants.n_init

        model_filename = 'model_data/mars-small128.pb'
        self.encoder = gdet.create_box_encoder(model_filename,batch_size=1)
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = DeepTracker(metric, max_age = max_age,n_init= n_init)

        self.capture = cv2.VideoCapture('Camcorder 2 Demo.mp4')
        if self.capture.isOpened():         # Checks the stream
            self.frameSize = (int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                               int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        Constants.SCREEN_HEIGHT = self.frameSize[0]
        Constants.SCREEN_WIDTH = self.frameSize[1]

        frame_width = int(self.capture.get(3))
        frame_height = int(self.capture.get(4))
   
        size = (frame_width, frame_height)
        self.result_video = cv2.VideoWriter('Camcorder 2 Demo_output.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         25, size)


    def getCurrentFrameAsImage(self):
            frame = self.currentFrame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pgImg = pygame.image.frombuffer(frame.tostring(), frame.shape[1::-1], "RGB")
            return pgImg


    def run(self, no_frames):
        result, self.currentFrame = self.capture.read()

        if not result:
            print("Can't receive frame (stream end?). Exiting ...")
            return 0
        datum = op.Datum()
        datum.cvInputData = self.currentFrame
        self.openpose.emplaceAndPop(op.VectorDatum([datum]))

        keypoints, self.currentFrame = np.array(datum.poseKeypoints), datum.cvOutputData
        # print(np.shape(keypoints))
        # input()
        # Doesn't use keypoint confidence
        try:
            poses = keypoints[:,:,:2]
            # Get containing box for each seen body
            boxes = poses2boxes(poses)
        except Exception as e:
            print(e)
            poses = np.zeros([1,25,3])
            boxes = []
        print('No of poses:',len(boxes))
        boxes_xywh = [[x1,y1,x2-x1,y2-y1] for [x1,y1,x2,y2] in boxes]
        features = self.encoder(self.currentFrame,boxes_xywh)
        # print(features)

        nonempty = lambda xywh: xywh[2] != 0 and xywh[3] != 0
        detections = [Detection(bbox, 1.0, feature, pose) for bbox, feature, pose in zip(boxes_xywh, features, poses) if nonempty(bbox)]
        # Run non-maxima suppression.
        boxes_det = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes_det, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        # Call the tracker
        self.tracker.predict()
        self.tracker.update(detections)

        try:
            data_to_save = np.reshape(keypoints,[np.shape(keypoints)[0],-1])
            np.savetxt("save_csv\point"+str(no_frames)+".csv", data_to_save , fmt='%.6g', delimiter=",")
        except Exception as e:
            print(e)
            data_to_save = np.zeros([1,75])
            np.savetxt("save_csv\point"+str(no_frames)+".csv", data_to_save , fmt='%.6g', delimiter=",")

        track_id = []
        track_id_sort = []
        sort_boxes = {}

        for track in self.tracker.tracks:
            color = None
            if not track.is_confirmed():
                color = (0,0,255)
            else:
                color = (255,255,255)
            bbox = track.to_tlbr()
            # print("Body keypoints:")
            # print(track.last_seen_detection.pose)
            cv2.rectangle(self.currentFrame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),color, 2)
            # if no_frames == 450:
            #     print((int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])))
            cv2.putText(self.currentFrame, "%s"%(track.track_id),(int(bbox[0]), int(bbox[1])-5),0, 5e-3 * 200, (0,255,0),2) # - ts%s # ,track.time_since_update
            track_id_sort.append(track.track_id)
            sort_boxes[track.track_id] = {'x1':int(bbox[0]), 'y1':int(bbox[1]), 'x2':int(bbox[2]), 'y2':int(bbox[3])}
            cv2.putText(self.currentFrame, "%s"%(no_frames+1),(5, 25),0, 5e-3 * 200, (255,0,0),2)

        # print(boxes)
        boxes_dict = [{'x1':x1,'y1':y1,'x2':x2,'y2':y2} for [x1,y1,x2,y2] in boxes]
        # print(boxes_dict)
        # print(sort_boxes)

        for bb_ in boxes_dict:
            try:
                max_ = max(sort_boxes, key = lambda k: get_iou(bb_,sort_boxes[k]))
                # print(max_)
                track_id.append(max_)
            except Exception as e:
                print('=====================This is the frame==================', no_frames)
                print(e)
                print(sort_boxes)
                input()
        # input()

        arr_track_id = np.array(track_id)
        print('Track ids', arr_track_id)
        print('Track ids of sort', track_id_sort)
        np.savetxt("save_csv\_track"+str(no_frames)+".csv", arr_track_id , fmt='%.6g', delimiter=",")

        # if no_frames == 450:
        #     print(boxes_xywh)
        #     input()

        self.result_video.write(self.currentFrame)

        if cv2.waitKey(1) == ord('p'):
            input()
