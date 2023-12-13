# Model 2 MRQA
This folder contains Model 2 of the Synchrony analysis. <br>
Modules 0_PreprocessForAutomation, 1_FaceDetection, 2_Reaching-Detection are the same and have been run already.
<br><br>
Synchrony is checked using the Multidimensional Recurrence Quantification Analysis (https://www.frontiersin.org/articles/10.3389/fpsyg.2016.01835/full). <br>
The toolbox used to evaluate the scores and the images is implemented in Matlab, whereas the input files to run the Matlab toolbox are processed in python.
<br><br>
## Part one: Python code to produce MRQA analysis inputs
This section uses as input the .json combined files produces in the previous modules. <br>
**Code description:** <br>
**main_discarded_frames.py** <br>  
This script takes as inputs the .json combined file containing info about mom's and baby's faces and bodies. <br>
It checks which frames are ok to work with and which are not for each dataset following this criteria: <br>
- a frame should contain at least two people <br>
- there sohuld be a couple mom-baby in the frame <br>
- mom and baby selected keypoints must score a selected confidence level <br>
<br><br>
The scipt discarded the frames that are not good and outputs an excel file for each dataset containing the following information:<br>
- total number of frames <br>
- how many good frames <br>
- how many bad frames <br><br>
*This is used as preliminary analysis.* <br><br>

# main_input_data_process # 
This script outputs an excel file for each dataset that is going to be the input of the MRQA.
Each row of the excel file represents a frame of the analysed video and it gives information about x and y coordinates for the 
selected keypoints (i.e. Neck and Nose) for both mom and baby.
If the frames is a discarded one (does not fit the criteria previously explained) a line og NaNs is displayed.


# functions.py #
This file contains all the functions to run the two main scripts.


## Run the code and analyzed parameters ## 
There are serveral paraments that can be set and changed before running the analysis.
- path root
- excel file name
- keypoints analyzed
- confidence score


## Results ## 





### Part two: Matlab code to run the MRQA analysis ###

## Installation

## Code description ##

## How to run the code ## 

## Results ## 
