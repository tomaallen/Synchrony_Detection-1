## How to run detect_face.py?
Where is the code from?
The code is based on Yolov7 (https://github.com/WongKinYiu/yolov7).

How to setup the environment?
Nothing special. Simply create an Python conda environment, install the the latest pytorch, opencv-python, pandas. Use pip to install any missing packages.

What's the input?
Please see detect_face.py, just scroll down to the bottom and there is explanation for each arguments.

What's the output?
1. A folder as the Yolov7 raw output, including the detected faces's bounding box txt files and the example video with bounding boxes drawn in each frame.
2. A csv file, whose length equals the video's frame count. In each row, ["cx_mom", "cy_mom", "cx_baby", "cy_baby"] are recorded. c means center. cx cy means the coordinate of the center of the bounding boxes. If there is no faces detected in one frame, then the row will contain zeros.
