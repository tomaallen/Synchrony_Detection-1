# Synchrony_Detection

## To Do in new fork:
- clean up RD/src of excess files

- check gitattribute and gitignore files
	- check with local git etc.
	- need to clear up handling of .dll files
	- if you can get it into a local repo then rename 2_Reaching-Detection to 2_PoseDetection


## Installation
1. Install Anaconda
1. **Download** the GitHub repository Synchrony_Detection from my new fork
1. install cuda=11.7 on device from here https://developer.nvidia.com/cuda-11-7-0-download-archive
	- if different version of cuda used you will need to change pytorch version in requirements.txt
	- using pytorch>2.0 will result in face_detection using cpu only on windows
	- you should not need cudnn as it is included in openpose binaries for windows (but I have it installed anyway)
1. Create an environment e.g. named synchrony_detection
	- MUST INSTALL PYTHON V3.7.16
	- `conda create -n synchrony_detection python==3.7.16`
	- `conda activate synchrony_detection`
	- `pip install -r {your path}\SynchronyDetection\requirements.txt`
1. Download best.pt and add to Synchrony_Detection\\1_FaceDetection\\yolov7-main folder (available to download on OneDrive)
1. Download openpose gpu version from https://github.com/CMU-Perceptual-Computing-Lab/openpose/releases and save to local drive
1. Download 3rd party for 2021 and models from here https://github.com/CMU-Perceptual-Computing-Lab/openpose/issues/1602#issuecomment-641653411
	- unzip all, including subfolders e.g. using 7-zip
	- add these downloads to respective folders in openpose download
1. Copy the folders bin, include, lib and models from your openpose folder to Synchrony_Detection\2-Reaching_Detection
	- Replace any existing files


## Data Preparation 
### Trim Videos
Trim the videos to obtain only the required portion (STT task, PCI task, etc.)

### Files Location and Naming Constraint
Videos should be stored in a single flat folder. Naming of files should **include participant names at the least** in order to combine outputs from multiple cameras at the end of analysis. 
<br><br> Example: BLBR001_SyncCam, LP004_BabyCam, Bangladesh012_SyncCam. etc.


## Running Analysis

## 1. Face Detection 
Open an anaconda terminal and type:
- `conda activate synchrony_detection` <br><br>
- `cd {your path}\SynchronyDetection\1_FaceDetection` <br><br>
- `python automate_face.py --folder {your analysis folder}`

Run the script and wait for the results to be saved.

## 2. Reaching Detection
### Notice that this section can run in parallel with section 1.
Open a new anaconda prompt and type:
- `conda activate synchrony_detection` <br><br>
- `cd {your path}\SynchronyDetection\2_Reaching-Detection\src` <br><br>
- `python automate_pose.py --folder {your analysis folder}`

Run the script and wait for the results to be saved.

## 3. Combined Analysis
The combined analysis script does three things
1. Copies the face csv file to the pose output folder
1. Generates a combined json file in the pose output folder
1. Runs cross-correlation model 1.0 by calling file_input_run_plots and file_input_run_parameters
	- **This could be adapted to use newer models**

To run the combined analysis, open a new anaconda prompt and type:
- `conda activate synchrony_detection` <br><br>
- `cd {your path}\3_CombineFacePose` <br><br>
- `python combined_analysis.py --folder {your analysis folder}`

Results are saved to cross_corr_output\combined_results.csv
Information about whether each step of analysis has run can be found in analysis_info\analysis_info.csv
