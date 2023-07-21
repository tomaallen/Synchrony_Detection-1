# Python program to write
# text on video

from scipy.io import savemat, loadmat
import numpy as np
import cv2
from _0_data_constants import *
import xgboost as xgb

from_training = False
if from_training:
    X_train = loadmat(INPUT_FOLDER + PREFIX + MATLAB_TRAINING_INPUT_NAN)['reaching_training_input_data']
    y_train = loadmat(INPUT_FOLDER + PREFIX + MATLAB_TRAINING_OUTPUT_NAN)['reaching_training_label']
    X_test = loadmat(INPUT_FOLDER + PREFIX + MATLAB_TEST_INPUT_NAN_ADD_NAN)['reaching_test_input_data']
    y_test = loadmat(INPUT_FOLDER + PREFIX + MATLAB_TEST_OUTPUT_NAN)['reaching_test_label']

    X = np.vstack((X_train, X_test))
    y = np.vstack((y_train, y_test))
    print('shape:', X.shape, y.shape)
else:
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

    X = input_data
    # Add NaN to match dimension for stereo camera input
    an_array = np.empty(X.shape)
    an_array[:] = np.NaN
    X = np.hstack((an_array, X))

    print('shape:', X.shape)

xg_reg = xgb.XGBRegressor()
xg_reg.load_model(INPUT_FOLDER + XGB_MODEL_NAN)

cap = cv2.VideoCapture(INPUT_FOLDER + 'Camcorder 1_Eval room_STT_skeleton.avi')
# get vcap property
width = cap.get(3)  # float `width`
height = cap.get(4)  # float `height`
fps = cap.get(cv2.CAP_PROP_FPS)
print(width, height, fps)

fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
out = cv2.VideoWriter(INPUT_FOLDER + 'Camcorder 1_Eval room_STT_skeleton_prediction.avi', fourcc, fps,
                      (int(width), int(height)))

no_frame = 0
color = (0, 255, 255)
while (True):

    # Capture frames in the video
    ret, frame = cap.read()

    # describe the type of font
    # to be used.
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Use putText() method for
    # inserting text on video
    # print(X[no_frame,:])
    text = xg_reg.predict(X[no_frame:no_frame + 1, :])
    if np.isnan(X[no_frame, :]).any():
        text_nan = 'NaN'
    else:
        text_nan = ''
    if text[0] >= 0.4:
        color = (255, 255, 255)
    else:
        color = (0, 255, 255)
    text = str(text) + text_nan
    cv2.putText(frame,
                str(text),
                (int(width / 2 - 100), int(height - 50)),
                font, 1,
                color,
                2,
                cv2.LINE_4)

    # Display the resulting frame
    cv2.imshow('video', frame)
    out.write(frame)

    # creating 'q' as the quit
    # button for the video
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    no_frame += 1

# release the cap object
cap.release()
# close all windows
cv2.destroyAllWindows()
