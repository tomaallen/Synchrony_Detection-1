# Synchrony_Detection

## Installation
1. Install Anaconda
1. Download or clone the GitHub repository Synchrony_Detection
1. install cuda=11.7 on device from here https://developer.nvidia.com/cuda-11-7-0-download-archive (or cuda=11.8 also works)
	- if cuda<11.7 used you will need to change pytorch version in requirements.txt
	- using pytorch>2.0 will result in face_detection using cpu only on windows
1. Create an environment e.g. named synchrony_detection
	- `conda create -n synchrony_detection python==3.7.16`
	- `conda activate synchrony_detection`
	- `pip install -r {your path}\Synchrony_Detection\requirements.txt`
1. Download openpose gpu version
	- download: https://github.com/CMU-Perceptual-Computing-Lab/openpose/releases/download/v1.7.0/openpose-1.7.0-binaries-win64-gpu-python3.7-flir-3d_recommended.zip
 	- unzip the downloaded folder
1. Download models
	- download: https://drive.google.com/drive/folders/1TUGl__i7x7JJKWsMts-RhyYNeS8UAIr3?usp=sharing
 	- add best.pt to Synchrony_Detection\\1_HeadDetection\\yolov7-main folder
	- drag and drop the models folder into the unzipped openpose folder - this adds additional files to the existing models folder
1. Copy the folders bin, include, lib and models from your openpose folder to Synchrony_Detection\2_PoseDetection


## Preparation 
### Naming Constraint
Naming of PCI videos should *include participant names and time point at the least* in order to select the best camera angle for each session. This is implemented in 3.0_CombinedHeadPose>combined_analysis.py.

### Specifying folder location
Videos should be stored in a single flat folder. This can be achieved by using preprocess.py if needed.
	- **ESSENTIAL: Open settings.py and change FOLDER to your folder location.**

## 1. Face Detection 
Open an anaconda terminal and type:
- `conda activate synchrony_detection` <br><br>
- `cd {your path}\Synchrony_Detection\1_HeadDetection` <br><br>
- `python detect_face.py`

Run the script and wait for the results to be saved.

## 2. Reaching Detection
### Notice that this section can run in parallel with section 1.
Open a new anaconda prompt and type:
- `conda activate synchrony_detection` <br><br>
- `cd {your path}\Synchrony_Detection\2_PoseDetection\src` <br><br>
- `python detect_pose.py`

Run the script and wait for the results to be saved.

## 3.0. Combined Analysis
The combined analysis script does three things
1. Copies the face csv file to the pose output folder
1. Generates a combined json file in the pose output folder
1. Checks for the best camera angle with the most good frames

To run the combined analysis:
- Edit combined_analysis.py lines 112-113 to extract participant id and timepoint from your filenames (or Synapse metadata)
	- This is crucial in selecting the best camera angle from each session
Then open a new anaconda prompt and type:
- `conda activate synchrony_detection` <br><br>
- `cd {your path}\Synchrony_Detection\3.0_CombineHeadPose` <br><br>
- `python combined_analysis.py`

Information about whether each step of analysis has run can be found in {settings.FOLDER}\analysis_info\analysis_info.csv
Information about how good each camera angle is can be found in {settings.FOLDER}\analysis_info\data_quality.csv
The best cameras are listed in {settings.FOLDER}\analysis_info\best_cameras.csv

## 3.1. Model 1 - Cross-correlations
TODO: [Text about cross-correlations model and explanation that it only runs on the best cameras]
- `conda activate synchrony_detection` <br><br>
- `cd {your path}\Synchrony_Detection\3.1_Model1_CrossCorr` <br><br>
- `python run_model1.py {your_video_fps}`

## 3.2. Model 2 - MdRQA

## 3.3 Model 3 - Graph Networks
