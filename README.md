# Synchrony_Detection
[for LEAP 1kD performers only - please do not circulate or redistribute]

This repository contains software to extract body key-points from videos of mother-infant interactions and compute synchrony metrics on the resultant time-series. Steps 1 and 2 of the pipeline extract key-points from the video data and assign them to either the mother or infant. Step 3 computes various synchrony metrics from the time-series of pose data. The pipeline is optimised to run on a flat folder structure of videos and is capable of automatically finding the best cameras for a given session or timepoint. To run the extraction and synchrony models, run steps 1, 2 and 3.0 before running your model(s) of choice.

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
 	- add best.pt to Synchrony_Detection\\head_detection\\yolov7-main folder
	- drag and drop the models folder into the unzipped openpose folder - this adds additional files to the existing models folder
1. Copy the folders bin, include, lib and models from your openpose folder to Synchrony_Detection\pose_detection

Model 2 specific:
- Download and install MATLAB 2021b
	- This older version of MATLAB is required for matlab engine to be compatible with python 3.7
	- Install the 'Image Processing Toolbox' and 'Statistics and Machine Learning Toolbox' when asked


## Preparation 
### Naming Constraint
Naming of PCI videos should *include participant numbers and time point at the least* in order to select the best camera angle for each session/timepoint. This is implemented in synchrony_analysis\combined_analysis.py and synchrony_analysis\data_quality_check.py

### Specifying folder location
Videos should be stored in a single flat folder. This can be achieved by using preprocess.py if needed. <br>
**ESSENTIAL: Open settings.py and change FOLDER to your folder location.**

## 1. Face Detection 
Open an anaconda terminal and type:
- `conda activate synchrony_detection` <br><br>
- `cd {your path}\Synchrony_Detection\head_detection` <br><br>
- `python detect_face.py`

Run the script and wait for the results to be saved.

## 2. Reaching Detection
*Notice that this section can run in parallel with section 1.*
Open a new anaconda prompt and type:
- `conda activate synchrony_detection` <br><br>
- `cd {your path}\Synchrony_Detection\pose_detection\src` <br><br>
- `python detect_pose.py`

Run the script and wait for the results to be saved.

## 3.0. Combined Analysis
The combined analysis script does two things
1. Copies the head/face csv file to the pose output folder
1. Generates a combined json file in the pose output folder, with id-assigned skeletons
1. Performs an initial data quality check, generating a csv file indicating which key-points are detected in each frame

Open a new anaconda prompt and type:
- `conda activate synchrony_detection` <br><br>
- `cd {your path}\Synchrony_Detection\synchrony_analysis` <br><br>
- `python combined_analysis.py`

Information about whether each step of analysis has run can be found in {settings.ANALYSIS_FOLDER}\analysis_info.csv. A basic summary of video quality for each recording can be found in {settings.ANALYSIS_FOLDER}\data_quality.csv. The initial data quality check results are saved in {settings.ANALYSIS_FOLDER}\analysis_info\frame_checks.

### Before running any models: Data Quality Check
- Edit get_ppt() and get_tp() in data_quality_check.py to extract participant id and timepoint from your filenames
	- This is crucial in selecting the best camera angle from each session
	- If you do not have multiple session/timepoints, make get_tp() return '1' for all videos
- Each model has a different data quality check depending on the model parameters and which key-points are utilised
	- The data quality check for models 1 and 2 are implemented with using synchrony_analysis\data_quality_check.py
- The best quality cameras from each session are listed in {settings.ANALYSIS_FOLDER}\best_cameras.csv. Analysis will be run on these cameras only.

## 3.1. Model 1 - Cross-correlations

### Method:
Model 1 implements a sliding windowed peak-pciking cross-correlational model on head and arm angles of the mother-infant dyad. A sliding cross-correlational model with peak-picking model was used by Klein et al. (2020) to investigate face-to-face mother-infant interactions. Like Klein et al., we investigate the cross-correlations between head and arm angles of mother and infant. However, we generalise the use of the model to more naturalistic, free-form parent-child interactions. Moreover, we calculate 4 interpretable metrics from the output of the cross-correlational model. The model considers linear relationships between mother and infant posture over time and is able to reveal who is leading postural interactions through the calculation of peak lag.

Klein, L., Ardulov, V., Hu, Y., Soleymani, M., Gharib, A., Thompson, B., ... & Matarić, M. J. (2020, October). Incorporating measures of intermodal coordination in automated analysis of infant-mother interaction. In Proceedings of the 2020 International Conference on Multimodal Interaction (pp. 287-295). https://doi/pdf/10.1145/3382507.3418870

### Metrics:
Pose Synchrony Model 1 outputs consists of 4 metrics

- Infant and Mother Initiated Interactions are calculated as the normalized number of mother initiations per unit time (ratio between no. mother initiation / length of the interaction).<br><br>
- Change of leader = normalized number of leading changes per unit time (ratio between # leading changes / length of the interaction). A lead change is when the individual leading/initiating the interaction has been switched between mother and infant from one window to the next. It is normalised the same as the above.<br><br>
- Synchrony Stability (intensity) = normalized correlation intensity (ratio between mean lag variance  / length of the interaction). The correlation intensity is defined as the variance in the peak lag, where the peak lag is calculated for each window.<br>

### Run Model 1:
- `conda activate synchrony_detection` <br><br>
- `cd {your path}\Synchrony_Detection\synchrony_analysis\cross_correlations` <br><br>
- `python run_model1.py {your_video_fps}`

Results are saved in settings.MODEL1_FOLDER.

### Implementation notes:
A data quality check is performed automatically to select the best cameras at each timepoint. For model 1, we run a data quality check for each pair of key-points used in angle calculations: ['LShoulder', 'LElbow'], ['RShoulder', 'RElbow'], ['Nose', 'LEar'], ['Nose', 'REar']. For each pair, we calculate the proportion of frames which contain both key-points for mother and infant. The final quality score is the mean of these four proportions. The best camera, with the highest quality score, is selected for each session/timepoint. The cross-correlations model is run for the best cameras only.

The model is implemented as follows:<br>
	- For each segment of good quality data (minimum 10 seconds), the model performs a sliding-windowed cross-correlation between mother and infant head angles. Window size was set to 3 seconds. Correlation strength and lag are calculated for each window. The same is carried out for arm angles.<br>
	- Mother/Infant initiated interactions, change of leaders and synchrony stability are calculated as described above, computed as a weighted mean across all segments (weighted for segment length). Final units are normalised per second.

## 3.2. Model 2 - MdRQA

### Method:
For Model 2, a Multi-dimensional Recurrence Quantification Analysis (MdRQA - Wallot, Roepstorff and Monster, 2016) approach has been implemented. MdRQA is a multivariate extension of simple RQA, which is an analysis technique that was developed to characterize the behavior of time-series that are the result of multiple interdependent variables, potentially exhibiting nonlinear behavior over time. Whilst model 1 considers linear relationships between mother and infant pose signals, MdRQA offers a complementary approach by taking into account non-linear contingencies between maternal and infant pose and movement. MdRQA is a recurrence-based analysis technique to gauge the coordination pattern of multiple variables over time and quantify the dynamic characteristics of a multivariate system. The basis of the MdRQA approach is phase-space reconstruction through time-delayed embedding. A phase-space is a space in which all possible states of a system under study can be charted.

Wallot, S., Roepstorff, A., & Mønster, D. (2016). Multidimensional Recurrence Quantification Analysis (MdRQA) for the analysis of multidimensional time-series: A software implementation in MATLAB and its application to group-level data in joint action. Frontiers in psychology, 7, 224211. https://doi.org/10.3389/fpsyg.2016.01835 <br>

### Metrics:
In our application of MdRQA, we focus our analysis on nose and neck key-points, as we consider these important indicators of adult-infant engagement and dynamics. Two key metrics are generated by this analysis:

1. Position recurrence - recurrence in the x-y positions of mother and infant key-points
1. Velocity recurrence - recurrence in the velocities of mother and infant key-points

### Run Model 2:
- `conda activate synchrony_detection` <br><br>
- `cd {your path}\Synchrony_Detection\synchrony_analysis\mdrqa` <br><br>
- `python run_model2.py {your_video_fps}` e.g. `python run_model2.py 30` <br><br>

NOTE: participant and timepoint are currently not provided in the MdRQA output by default, just filename.

### Implementation notes:
- A data quality check is performed for each video based upon the number of frames in which mother and infant nose and neck are detected. The best video, with the most good frames, is selected for each session/timepoint.
- The rest of the MdRQA model is implemented in MATLAB (this may change to Python in future updates):
	1. Using the results from the data quality check, frames in which the detection confidence of any keypoint is below threshold (<0.3) are ignored.
	1. The time series of mother and infant nose/neck positions is smoothed using a Savitzky-Golay filter.
	1. Instantaneous velocities are calculated for every other frame of the video.
	1. Lag and embedding dimensions hyper-parameters are optimised algorithmically (see mdrqa\run_mdrqa.m and mdrqa\mdembedding-master for details).
	1. MdRQA is performed on both the position and velocity time-series. See run_model3.py for MdRQA hyper-parameters.
	1. Results are saved in settings.MODEL2_FOLDER.

Savitzky, A., & Golay, M. J. (1964). Smoothing and differentiation of data by simplified least squares procedures. Analytical chemistry, 36(8), 1627-1639.


## 3.3 Model 3 - Graph Networks

## Method:
Model 3 uses the concept of transfer entropy as a statistical metric to create a connectivity network between the infant and mother's key-point velocities. Transfer entropy determines the amount of information (asymmetric) transferred between two processes/individuals. The model creates a graph of transfer entropy between infant and mother's nose, neck, wrist, and elbow, since these key-points contain the most significant and reliable information for analysis. The density and strength of the network are then extracted, based on graph theory, to obtain a quantitative measure of the network architecture. These metrics give an idea of how many and how solid the connections are between different key-points throughout the graph network.

## Metrics:
Model 3 produces two directional metrics: strength and density. Both mother-to-infant and infant-to-mother are calculated for each metric.

1. Strength - the sum of all connection weights in the pose graph network
1. Density - the proportion of connections above threshold in the pose graph network

## Run Model 3:
- `cd {your path}\synchrony_analysis\graph_network` <br><br>
- `python run_model3.py {your_video_fps}` e.g. `python run_model3.py 30` <br><br>

 ## Implementation Notes
Model 3 does not currently use the data quality check to select the best camera angle, but uses an epoch-by-epoch check instead. Results can be mean averaged for each participant/timepoint in post-processing. The script automatically separates the data into batches of size n - this step is required for running permutations. We recommend manually calling main() in run_model3.py for the last batch if this batch contains fewer than 90 videos.

The main Model 3 script is run_model3.py and it can be divided into 3 sections for an easier analysis: <br>
### 1.	Epoch Check – retrieves the number of good epochs keeping in mind that:

good frame = frame containing mom-baby and pose detections with an acceptable confidence score <br>
epoch = set of continuous good frames <br>
good epoch = epoch with at least 3 seconds of good frames <br>

This number is strongly dependent on the recording’s fps (i.e. the threshold will be set to >90  for 3 seconds consecutive good frames for fps = 30).
The result of this first section is a .xlsx file pointing out, for each file, the # of Total frames, # of Discarded frames, # of Good frames, # Total epochs, and # Good epochs (for 3/5 seconds). <br>

### 2.	Transfer Entropy and P-Values – evaluates adjacency matrices with the transfer entropy method, creates the permutation sets for the normal distributions, and computes the P-Values.

#### 2.1  
Infant’s and mother’s key-point velocities are computed for selected key-points (nose, neck, right wrist, right elbow, left wrist, left elbow) using key-points positions retrieved from the JSON files (modules 1, 2, 3.0). <br>
Folders called baby_velocities and mom_velocities are automatically created, containing .txt velocities files for each participant file analysed.
Each .txt file is a list of velocities divided by key point and by epoch. <br>

#### 2.2
Adjacency matrices (6x6 matrices, one for each epoch) with the transfer entropy method are now created. <br>
Each element of the matrix is computed by applying the transfer entropy formula between one mom’s velocity vector and one baby’s velocity vector. <br>

Fig.1 shows how the matrix is composed: the x-axis stores the mom’s key-points velocities values, and the y-axis stores the baby’s ones. <br>
Transfer entropy adjacency matrices have been evaluated for both baby-mom and mom-baby directions. <br>
Folders called baby-mom_te and mom-baby_te are automatically created, containing .txt files with all the possible combinations of matrices among the participants' files stored in the root directory.  <be>
<img src="https://github.com/isabella-sole/Synchrony_Detection/blob/main/te%20matrix.png" alt="" width="650" height="650">


Fig. 1

#### 2.3
Normal distributions and P-values are built for each batch. Batches are automatically assigned by the script. <br>
Permutation sets  are created using the Partner shuffling technique (one permutation set contains the transfer entropy adjacency matrix of a babyN-momN  couple + the transfer entropy adjacency matrices of all the remaining combinations of couples not containing babyN-momN). <br>

The random distribution is now evaluated using the mean and standard deviation of the permutation sets. P-values can now be computed. <br>

Folders called baby-mom_subsets and mom-baby_subsets are automatically created and they contain all the permutations sets. <br>
baby-mom_pvalues and mom-baby_pvalues folders are also created containing P-values matrices (1 matrix for each epoch, only 3 epochs per participant are considered) for each participant file. <br>


### 3.	Graph Parameters – Strength and Density are retrieved.
For each participant file, 3 graphs (for the 3 epochs contained in the .txt file) are built and the graph metrics of Strength and Density are computed. <br>
In each graph the body key-points represent the graph nodes and the connectivity values (weighted graph) are the graph edges.
All the matrices are not symmetrical but directional. <br>
The final outputs are two .txt/.csv files (GraphMetrics_baby-mom and GraphMetrics_mom-baby) containing the Density and Strength values for each epoch for each participant couple.




