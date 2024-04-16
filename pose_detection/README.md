# Reaching-Detection
Draft version...

## Some introductions
The pose estimation and tracking in this project are based on two methods:
- [Openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose): for multi-person pose estimation.
- [DeepSORT](https://github.com/nwojke/deep_sort): for multiple object tracking.

## How to run the code on a video (image) or some videos (images):
- **_Step 1_**: Download the code in this project. The code will be saved in _reaching-detection_ folder.
- **_Step 2_**: Download Openpose pretrained models. Further instructions for this step can be found in the later part of this README. The downloaded data (after unzipping) will be saved in _openpose_ folder.
- **_Step 3_**: Copy the folders _bin, include, lib, models_ in the _openpose_ folder to the _reaching-detection_ folder:
![](https://github.com/NTU-LEAP-1kD/Reaching-Detection/blob/main/openpose_to_reaching_detection.png)
- **_Step 4_**: Set up the python environment for running the model. See detailed instructions for this step in the later part of this README.
- **_Step 5_**: To start the pose detection and tracking, open a command window inside the _src_ folder, activate the newly created environment in Step 4, and then run the main file **main.py** by calling `python main.py` in the command window.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; To run on a video (image) or a folder containing some videos (and/or images), call `python main.py --input "path to file or folder"`.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Noting that calling `python main.py` without an argument will run the model on the default input folder _data\list_of_files_.
- **_Step 6_**: The output files for pose estimation and tracking can be found in _data\output_files\\_.


### Instructions for **_Step 2_** - downloading necessary Openpose pretrained models:
1. (For computer with Nvidia GPU) For maximum speed, you should use OpenPose in a machine with a Nvidia GPU version. If so, you must upgrade your Nvidia drivers to the latest version (in the Nvidia "GeForce Experience" software or its [website](https://www.nvidia.com/Download/index.aspx)).
2. **Download the latest OpenPose version from the [Releases](https://github.com/CMU-Perceptual-Computing-Lab/openpose/releases) section.**
3. **Follow the Instructions.txt file inside the downloaded zip file to download the models required by OpenPose (about 500 Mb).**
4. (Optional) Then, you can run OpenPose from the PowerShell command-line by following [doc/01_demo.md](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/01_demo.md).

Further instructions for **_Step 2_** can be found via [Windows Portable Demo](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/installation/0_index.md#windows-portable-demo).

### Instructions for **_Step 4_** - setting up a new python environment:

The new environment must use python 3.7 (same python version as the released openpose models). The sequential commands (to be run on a Command Prompt or an Anaconda Prompt) for creating, activating a new environment named _reaching_env_ and then installing required packages are shown as follows:

`conda create -n reaching_env python=3.7`

`conda activate reaching_env`

`pip install -r requirements.txt`


