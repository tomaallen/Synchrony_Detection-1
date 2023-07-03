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
Open a terminal and type:
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

Open the script **main.py** and edit at line 54 --input with the path of the folder containing the videos you want to analyse and line 58 --output with the folder path where you want to store the results.
Run the script by typing in the anaconda prompt:

## 3. Create the Json Combined file
scrivi domani


