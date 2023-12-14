# Model 2 MRQA
This folder contains Model 2 of the Synchrony analysis. <br>
Modules 0_PreprocessForAutomation, 1_FaceDetection, 2_Reaching-Detection are the same and have been run already.
<br><br>
Synchrony is checked using the Multidimensional Recurrence Quantification Analysis (https://www.frontiersin.org/articles/10.3389/fpsyg.2016.01835/full). <br>
The toolbox used to evaluate the scores and the images is implemented in Matlab, whereas the input files to run the Matlab toolbox are processed in python.
<br><br>
# Part One: Python code to produce MRQA analysis inputs
This section uses as input the .json combined files produces in the previous modules. <br>
## Code description: <br>
**main_discarded_frames.py** <br>  
This script takes as inputs the .json combined file containing info about mom's and baby's faces and bodies. <br>
It checks which frames are ok to work with and which are not for each dataset following this criteria: <br>
- a frame should contain at least two people <br>
- there sohuld be a couple mom-baby in the frame <br>
- mom and baby selected keypoints must score a selected confidence level <br>
<br>
The scipt discarded the frames that are not good and outputs an excel file for each dataset containing the following information:<br>
- total number of frames <br>
- how many good frames <br>
- how many bad frames <br><br>
*This is used as preliminary analysis.* <br><br>

**main_input_data_process.py** <br>
This script produces an excel file for each dataset that is going to be the input of the MRQA. <br>
Each excel file row represents a video frame and the columns are the selected kwypoints x and y coordinates for (i.e. Neck and Nose) for both mom and baby. <br>
If the frames is a bad one (does not fit the criteria previously explained) a line of NaNs is displayed. <br>
<br>
**functions.py** <br>
This file contains all the functions to run the two main scripts. <br>
<br>
## Run the code 
**Preliminary analysis for your dataset:**
- open the script *python main_discarded_frames.py* and change the **rootdir** parameter with the path to the folder where all the datasets are stored and change the **workbook** parameters with the name you want to assign to the excel file <br><br>
- `cd pathToYourFolder`<br><br>
- `python main_discarded_frames.py` <br><br>

**MRQA input preparation:**
- open the script *python main_input_data_process.py* and change the **rootdir** parameter with the path to the folder where all the datasets are stored.<br><br>
- `cd pathToYourFolder`<br><br>
- `python main_discarded_frames.py` <br><br>

## Results  
The *python main_discarded_frames.py* script produces one Excel file with the results of all the datasets analyzed.<br>
Each row of the file contains the total amount of frames, the # of good frames, and the # of bad frames of the analyzed dataset. <br> 
<br>
The *python main_discarded_frames.py* script produces one Excel file for each dataset.<br>
Each row of the file contains the x and y coordinates of the selected key points (for both mom and by) in a specific frame.


# Part two: Matlab code to run the MRQA analysis ###

## Installation

## Code Description 

## How to run the code 

## Results 
