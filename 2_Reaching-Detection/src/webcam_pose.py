# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse
import time
import numpy as np

try:
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

    # Flags
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", default="../examples/media/COCO_val2014_000000000428.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
    args = parser.parse_known_args()

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = "../models/"

    # Add others in path?
    for i in range(0, len(args[1])):
        curr_item = args[1][i]
        if i != len(args[1])-1: next_item = args[1][i+1]
        else: next_item = "1"
        if "--" in curr_item and "--" in next_item:
            key = curr_item.replace('-','')
            if key not in params:  params[key] = "1"
        elif "--" in curr_item and "--" not in next_item:
            key = curr_item.replace('-','')
            if key not in params: params[key] = next_item

    # Construct it from system arguments
    # op.init_argv(args[1])
    # oppython = op.OpenposePython()

    cap = cv2.VideoCapture('video.avi')
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    no_frames = 0
    t0 = time.time()

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
   
    size = (frame_width, frame_height)
    result = cv2.VideoWriter('video_output.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         25, size)
    while True: # and no_frames < 5
        # Capture frame-by-frame
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # Process Image
        datum = op.Datum()
        imageToProcess = frame  # cv2.imread(args[0].image_path)
        datum.cvInputData = imageToProcess
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))

        # Display Image
        # print("Body keypoints: \n" + str(datum.poseKeypoints))
        result.write(datum.cvOutputData)
        # cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", datum.cvOutputData)
        # print(np.shape(datum.poseKeypoints))
        try:
            shape_1 = (np.shape(datum.poseKeypoints)[0])
            data_to_save = np.reshape(datum.poseKeypoints,[shape_1,-1])
            np.savetxt("save_csv\point"+str(no_frames)+".csv", data_to_save , fmt='%.6g', delimiter=",")
        except Exception as e:
            print(e)
            data_to_save = np.zeros([1,75])
            np.savetxt("save_csv\point"+str(no_frames)+".csv", data_to_save , fmt='%.6g', delimiter=",")
        # cv2.waitKey(0)
        # Display the resulting frame
        # cv.imshow('frame', gray)
        no_frames += 1
        if cv2.waitKey(1) == ord('q'):
            break
        print("Number of frames so far:", no_frames)
    print('Total time:', time.time() - t0)

except Exception as e:
    print(e)
    sys.exit(-1)
