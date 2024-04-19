# Synchrony_Detection
[for LEAP 1kD performers only - please do not circulate or redistribute]

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
Naming of PCI videos should *include participant numbers and time point at the least* in order to select the best camera angle for each session. This is implemented in synchrony_analysis\combined_analysis.py.

### Specifying folder location
Videos should be stored in a single flat folder. This can be achieved by using preprocess.py if needed.
	- **ESSENTIAL: Open settings.py and change FOLDER to your folder location.**

## 1. Face Detection 
Open an anaconda terminal and type:
- `conda activate synchrony_detection` <br><br>
- `cd {your path}\Synchrony_Detection\head_detection` <br><br>
- `python detect_face.py`

Run the script and wait for the results to be saved.

## 2. Reaching Detection
### Notice that this section can run in parallel with section 1.
Open a new anaconda prompt and type:
- `conda activate synchrony_detection` <br><br>
- `cd {your path}\Synchrony_Detection\pose_detection\src` <br><br>
- `python detect_pose.py`

Run the script and wait for the results to be saved.

## 3.0. Combined Analysis
The combined analysis script does three things
1. Copies the head/face csv file to the pose output folder
1. Generates a combined json file in the pose output folder, with id-assigned skeletons

Then open a new anaconda prompt and type:
- `conda activate synchrony_detection` <br><br>
- `cd {your path}\Synchrony_Detection\synchrony_analysis` <br><br>
- `python combined_analysis.py`

Information about whether each step of analysis has run can be found in {settings.FOLDER}\analysis_info\analysis_info.csv
A basic summary of camera angle quality can be found in {settings.FOLDER}\analysis_info\data_quality.csv

**Before running any models:**
- Edit get_ppt() and get_tp() in data_quality_check.py to extract participant id and timepoint from your filenames (or Synapse metadata)
	- This is crucial in selecting the best camera angle from each session
- Each model has a different data quality check depending on the model parameters
	- All are implemented with data_quality_check() in data_quality_check.py
- The best cameras from each session are listed in {settings.FOLDER}\analysis_info\best_cameras.csv

## 3.1. Model 1 - Cross-correlations
Pose Synchrony Model 1 data consists of 4 metrics:

- Infant and Mother Initiated Interactions are calculated as the normalized number of mother initiations per unit time (ratio between no. mother initiation / length of the interaction). It is calculated like so:

	For each segment of data, perform a sliding-windowed cross-correlation. 
	For each segment, find the number of windows in which peak lag indicates mother or infant was leading the interaction (+ve or -ve).
	Sum the number of mother/infant led windows (interactions) across all segments in the video and divide by the total length of all segments combined.
	Final result is normalised per second.

- Change of leader = normalized number of leading changes per unit time (ratio between # leading changes / length of the interaction). A lead change is when the individual leading/initiating the interaction has been switched between mother and infant from one window to the next. It is normalised the same as the above.

- Synchrony Stability (intensity) = normalized correlation intensity (ratio between mean lag variance  / length of the interaction). The correlation intensity is defined as the variance in the peak lag, where the peak lag is calculated for each window.

To run model 1:
- `conda activate synchrony_detection` <br><br>
- `cd {your path}\Synchrony_Detection\synchrony_analysis\cross_correlations` <br><br>
- `python run_model1.py {your_video_fps}`

## 3.2. Model 2 - MdRQA

## 3.3 Model 3 - Graph Networks


## Model Overview
Model 3 uses the concept of transfer entropy as a statistical metric to create a connectivity network between the baby and mom’s key point velocities. <br>
Transfer entropy determines the amount of information (asymmetric) transferred between two processes.  Afterwards, the density and strength metrics are extracted, based on graph theory, to obtain a quantitative measure of the network architecture. <br>
These metrics will give an idea of how many and how solid the connections are between different key points on average. <br>
The model focuses on the nose, neck, right wrist, right elbow, left wrist, and left elbow key points, since they contain the most significant and reliable information for the analysis. <br>
Model 3 data input are the JSON files retrieved from the head and body detection (points 1, 2 and 3.0). <br>

## How to run the code
Navigate inside the folder synchrony_analysis:
- `cd “C:\your path to\synchrony_analysis\graph_network”` <br><br>
Launch transfer_entropy_connectivity_network.py script by typing the root directory path (where the JSON files have been saved, to be used as inputs), the base directory path (where you want to save the output of model3 analysis) and specifying the recordings fps: <br>
i.e.: <br>
- `python run_model3.py “C:\your\root\directory\” “C:\your\base\path” 30` <br><br>
Since the process is completely automatised and will run over all the participants' folders that have to be analysed, it is fundamental to have a maximum of 30 participant folders (each of them will have 2/3 recordings) inside the root directory. <br>
Too many participant folders inside the root directory will exponentially increase the computational time due to the elevated number of permutation sets created (explained in detail in the Scripts and Output Explanation section). <br>
If more than 30 participants have to be analysed, split the analysis in different root folders containing <30 participants. <br>

 ## Scripts and Output Explanation

The main Model 3 script is  transfer_entropy_connectivity_network.py: it can be divided into 3 sections for an easier analysis: <br>
### 1.	Epoch Check – retrieves the number of good epochs keeping in mind that:

good frame = frame containing mom-baby and pose detections with an acceptable confidence score <br>
epoch = set of continuous good frames <br>
good epoch = epoch with at least 3 seconds of good frames <br>

This number is strongly dependent on the recording’s fps (i.e. the threshold will be set to >90  for 3 seconds consecutive good frames for fps = 30).
The result of this first section is a .xlsx file pointing out, for each file, the # of Total frames, # of Discarded frames, # of Good frames, # Total epochs, and # Good epochs (for 3/5 seconds). <br>

### 2.	Transfer Entropy and P-Values – evaluates adjacency matrices with the transfer entropy method, creates the permutation sets for the normal distributions, and computes the P-Values.

#### 2.1  
Baby’s and mom’s key points velocities are computed for selected key points (nose, neck, right wrist, right elbow, left wrist, left elbow) using key points positions retrieved from the JSON files (modules 1, 2, 3.0). <br>
Folders called baby_velocities and mom_velocities are automatically created, containing .txt velocities files for each participant file analysed.
Each .txt file is a list of velocities divided by key point and by epoch. <br>

#### 2.2
Adjacency matrices (6x6 matrices, one for each epoch) with the transfer entropy method are now created. <br>
Each element of the matrix is computed by applying the transfer entropy formula between one mom’s velocity vector and one baby’s velocity vector. <br>

Fig.1 shows how the matrix is composed: the x-axis stores the mom’s key points velocities values, and the y-axis stores the baby’s ones. <br>
Transfer entropy adjacency matrices have been evaluated for both baby-mom and mom-baby directions. <br>
Folders called baby-mom_te and mom-baby_te are automatically created, containing .txt files with all the possible combinations of matrices among the participants' files stored in the root directory.  <be>
<img src="https://github.com/isabella-sole/Synchrony_Detection/blob/main/te%20matrix.png" alt="" width="650" height="650">


Fig. 1

#### 2.3
Normal distributions and P-values are built. <br>
Permutation sets  are created using the Partner shuffling technique (one permutation set contains the transfer entropy adjacency matrix of a babyN-momN  couple + the transfer entropy adjacency matrices of all the remaining combinations of couples not containing babyN-momN). <br>

The random distribution is now evaluated using the mean and standard deviation of the permutation sets. P-values can now be computed. <br>

Folders called baby-mom_subsets and mom-baby_subsets are automatically created and they contain all the permutations sets. <br>
baby-mom_pvalues and mom-baby_pvalues folders are also created containing P-values matrices (1 matrix for each epoch, only 3 epochs per participant are considered) for each participant file. <br>


### 3.	Graph Parameters – Strength and Density are retrieved.
For each participant file, 3 graphs (for the 3 epochs contained in the .txt file) are built and the graph metrics of Strength and Density are computed. <br>
In each graph the body key points represent the graph nodes and the connectivity values (weighted graph) are the graph edges.
All the matrices are not symmetrical but directional. <br>
The final outputs are two .txt files (GraphMetrics_baby-mom and GraphMetrics_mom-baby) containing the Density and Strength values for each epoch for each participant couple.




