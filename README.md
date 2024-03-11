# Synchrony_Detection
## Installation
1. Install Anaconda from https://www.anaconda.com/download and follow the installation tutorial
1. Download the GitHub repository Synchrony_Detection 
1. Install cuda=11.7 on device from here https://developer.nvidia.com/cuda-11-7-0-download-archive (or cuda=11.8 also works)
	- if cuda<11.7 used you will need to change pytorch version in requirements.txt
	- using pytorch>2.0 will result in face_detection using cpu only on windows
1. Create an environment e.g. named synchrony_detection
	- `conda create -n synchrony_detection python==3.7.16`
	- `conda activate synchrony_detection`
	- `pip install -r {your path}\Synchrony_Detection\requirements.txt`
1. Download openpose gpu version
	- download: https://github.com/CMU-Perceptual-Computing-Lab/openpose/releases/download/v1.7.0/openpose-1.7.0-binaries-win64-gpu-python3.7-flir-3d_recommended.zip
 	- unzip the downloaded folder
1. Download models from Models link located here https://github.com/CMU-Perceptual-Computing-Lab/openpose/issues/1602#issuecomment-641653411
2. download best.pt from https://drive.google.com/drive/folders/1TUGl__i7x7JJKWsMts-RhyYNeS8UAIr3?usp=sharing
 	- add best.pt to Synchrony_Detection\\1_FaceDetection\\yolov7-main folder
	- drag and drop the models folder into the unzipped openpose folder - this adds additional files to the existing models folder
1. Copy the folders bin, include, lib and models from your openpose folder to Synchrony_Detection\2-Reaching_Detection


## Data Preparation 
### Trim Videos
Trim the videos to obtain only the required portion (STT task, PCI task, etc.)

### Files Location and Naming Constraint
Videos should be stored in a single flat folder. Naming of files should *include participant names and time point at the least* in order to combine outputs from multiple cameras at the end of analysis. 
<br><br> Example: BLBR001_SyncCam, LP004_BabyCam, Bangladesh012_SyncCam. etc.


## Running Analysis

## 1. Face Detection 
Open an anaconda terminal and type:
- `conda activate synchrony_detection` <br><br>
- `cd {your path}\Synchrony_Detection\1_FaceDetection` <br><br>
- `python automate_face.py --folder {your analysis folder}`

Run the script and wait for the results to be saved.

## 2. Reaching Detection
### Notice that this section can run in parallel with section 1.
Open a new anaconda prompt and type:
- `conda activate synchrony_detection` <br><br>
- `cd {your path}\Synchrony_Detection\2_Reaching-Detection\src` <br><br>
- `python automate_pose.py --folder {your analysis folder}`

Run the script and wait for the results to be saved.

## 3. Combined Analysis & Model 1
The combined analysis script does three things
1. Copies the face csv file to the pose output folder
1. Generates a combined json file in the pose output folder
1. Runs cross-correlation model 1.0 by calling file_input_run_plots and file_input_run_parameters
	- **This could be adapted to use newer models**

To run the combined analysis, open a new anaconda prompt and type:
- `conda activate synchrony_detection` <br><br>
- `cd {your path}\Synchrony_Detection\3_CombineFacePose` <br><br>
- `python combined_analysis.py --folder {your analysis folder} --fps {your fps}`

Results are saved to cross_corr_output\combined_results.csv
Information about whether each step of analysis has run can be found in analysis_info\analysis_info.csv

## 4. Combining Results Across Cameras
If you have run the analysis on multiple cameras per participant you should combine the outputs as follows to give a single estimate of pose synchrony for each participant...
