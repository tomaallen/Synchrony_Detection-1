# Model 2 MRQA
This folder contains Model 2 of the Synchrony analysis.
Modules 0_PreprocessForAutomation, 1_FaceDetection, 2_Reaching-Detection are the same and have been run already.

This time, Synchrony is checked using the Multidimensional Recurrence Quantification Analysis toolbox.
Description of the toolbox...

#### Part one: Python code to prodcue the inputs for the MRQA analysis ###
At this point, modules  have been already run.
This section uses as input the .json combined files produces by the previous modules.

## Code description ##
The folder contains 3 main scripts:
- main_discarded_frames.py
- main_input_data_process.py
- functions.py

# main_discarded_frames.py # 
This algorithm takes as inputs the .json combined file that contains for each participant information about mom and baby faces 
and bodies.
This scipt checks which frames are ok to work with and which are not for each dataset.
The criteria to estabilish when a frame is a good one are the following:
- a frame should contain at least two people (mom and least, experimenters, etc.)
- there sohuld be a couple mom-baby in the frame (since we have to check synchrony between them)
- mom and baby selected keypoints have to have a certain confidence score that can be choosen

The scipt discarded the frames that are not good and outputs an excel files that says for each datasets how many good and bad 
frames we have. This is used as preliminary analysis.

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
