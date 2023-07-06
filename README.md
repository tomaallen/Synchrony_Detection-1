# Synchrony_Detection

## Installation
Install the required packages and create an anaconda environment

## Data Preparation 
### Trim Videos
.... Angshuk code

### Files Location and Naming Constraint
Create a main folder named **To_analyse** and inside it other three folders called **SyncCam** **BabyCam** **MomCam**. <br><br>
Store each recording that you want to analyze inside one of the previous folders, depending on the POV you are using. <br><br>
The video files **MUST** be named in the following way (if they are not the algorithm is not going to work):
- Location identifier and participant number
- Underscore **_**
- Camera used
<br><br> Example: BLBR001_SyncCam, LP004_BabyCam, Bangladesh012_SyncCam. etc.

## 1. Face Detection 
Open an Anaconda terminal and type:
- `conda activate synchrony_detection` <br><br>
- `cd C:\Users\isabellasole.bisio\Desktop\SynchronyDetection\1_FaceDetection` <br><br>
- `jupyter notebook` <br><br>

Open **Automate_face_pose_script_conda_env.ipynb** and edit: 
- **folder_path** with the path of the **To_analsyse** folder
- **save_path** with the path of the folder where you want to store the results

Run the script and wait for the results to be saved.

## 2. Reaching Detection
### Notice that you can 2. in parallel with 1.
Open an anaconda propt and type:
- `conda activate synchrony_detection` <br><br>
- `cd C:\Users\isabellasole.bisio\Desktop\Synchrony_Detection\2_Reaching-Detection\src` <br><br>

Open the script **main.py** and edit at line 54 --input with the path of the folder containing the videos you want to analyze and line 58 --output with the folder path where you want to store the results.
Run the script by typing in the anaconda prompt:

## 3. Create the Json Combined file
### Copy csv Face
- Go inside _C:\Users\youruser\Desktop\SynchronyAnalysis\json_combined_ and open _copy_csv_face.py_
- Edit the following:<br><br>
1. _--input_ copy here the path of the folder containing all the Face Detection output, ex. _Y:\Synchronised_trimmed_videos_STT\Face_detect_output\Baby_cam_. <br>
Note that nside the last folder of the path (in this example _Baby_cam_), there MUST all the participant folders <br><br>
2. _--output_ copy here the path of the Pose Detection Output, ex. _Y:\Synchronised_trimmed_videos_STT\Pose_detect_output\Baby_Cam _ <br><br>
3. Edit _csv_face_folder_ such that the algorithm will be able to extract only the name of the participant (ex. BLBR006) and discard the rest.
- Save the script and run the Anaconda terminal and type `cd C:\Users\labadmin\Desktop\SynchronyAnalysis-Upload\json_combined` <br><br>
- `python copy_csv_face.py` <br><br>

### Combine Face Detection and Body Pose Outputs 
- Go inside _C:\Users\youruser\Desktop\SynchronyAnalysis\json_combined_ and open _2_z_per_frame_id__use_yolo_face.py_
- Edit the following:<br><br>
_--reach_dir_ copy here the path of the Pose Detection output with csv face, ex. _Y:\Synchronised_trimmed_videos_STT\Pose_detect_output\Baby_Cam_
- Save the script and run the Anaconda terminal and type `cd C:\Users\labadmin\Desktop\SynchronyAnalysis-Upload\json_combined` <br><br>
- `python 2_z_per_frame_id__use_yolo_face.py` <br><br>

### Run analysis plots
- Go inside _C:\Users\youruser\Desktop\SynchronyAnalysis\3_PCI-Analysis_ and open _run_analysis_plots.py_
- Edit the following:<br><br>
1. _--PCI_dir_ copy here the path of the PCI Analysis Folder, ex. _C:\Users\youruser\Desktop\SynchronyAnalysis\3_PCI-Analysis_ <br><br>
2. _--reach_dir_ copy here the path of the folder containing the Pose Detection outputs, ex. _Y:\Synchronised_trimmed_videos_STT\Pose_detect_output\Baby_Cam_ <br><br>
3. _participant_name_ adapt it to the name of the files <br><br>
- Save the script and run the Anaconda terminal and type `cd C:\Users\labadmin\Desktop\SynchronyAnalysis-Upload\3_PCI-Analysis` <br><br>
- `python run_analysis_plots.py` <br><br>

### Run analysis parameters
- Go inside _C:\Users\youruser\Desktop\SynchronyAnalysis\3_PCI-Analysis_ and open _run_analysis_parameters.py_
- Edit the following:<br><br>
1. _--PCI_dir_ copy here the path of the PCI Analysis Folder, ex. _C:\Users\youruser\Desktop\SynchronyAnalysis\3_PCI-Analysis_ <br><br>
2. _--reach_dir_ copy here the path of the folder containing the Pose Detection outputs, ex. _Y:\Synchronised_trimmed_videos_STT\Pose_detect_output\Baby_Cam_ <br><br>
3. _participant_name_ adapt it to the name of the files <br><br>
- Save the script and run the Anaconda terminal and type `cd C:\Users\labadmin\Desktop\SynchronyAnalysis-Upload\3_PCI-Analysis` <br><br>
- `python run_analysis_parameters.py` <br><br>






