# PCI Synchrony Detection

## Requirements
This code can be run with the python 3.7 version, and the following python libraries:
- numpy
- scipy

## Getting Started

Clone the repository with the following command:
```
git clone https://github.com/micolspitale93/LEAP_PCI.git
```

Then, install the requirements with the following command:

```
cd LEAP_PCI
pip install -r requirements.txt
```

The repository contains an example folder which includes Open-pose files extracted (from a public available dataset reported in this paper) of mother and infant interaction. 

Then the following scripts are included in this folder:
- json2csv.py - utils function to convert the output of the json files (combined) into a csv file
- corr_functions.py - script that contains the functions needed to compute the cross-correlation parameters and windowed sliding window peack-picking CC
- preprocess.py - script for extracting features from mother and infant skeleton (e.g., arm and head angles)
- visualization.py - script to visualize the animated skeleton from csv file (from Thiet code)
- run_analysis_parameters.py - run the analysis to extract the synchrony parameters
- run_analysis_plots.py - run the analysis to visualize the CC peak curves 

After that, create a new folder which contains the datasets as follows:
```
mkdir Openpose outputs
cd Openpose outputs
mkdir Bangladesh Data
mkdir Brazil Data
mkdir Singapore Data
```

In each of this subfolder, copy and paste Thiet data with all the correspoding subfolders (e.g., ./Singapore Data/LP004_PCI/json_files). Do that for all the data that you want to analysis beloging to one of those sub-sets.

## Tutorial


### Running Model 1

This code allows you to run 2 different analysis:
- Getting analysis and plots of peak curve vs. lag of the CC analysis 
- Getting synchrony parameters

It is important that you ran the two analyses in sequence. 
First, you run the following command:
```
cd ./LEAP_PCI/
python run_analysis_plots.py
```
You will get a folder (which corresponds to each of the video analysed) that contained a json file (segmented_output.json). This json files contains all the parameters (e.g., number of initiations from the mother) but also the time-based peak_indices, and peak_correlations parameters (array with multiple values, each of one corresponds to a specific time-window). It includes also the start_time and end_time indicating its time interval, feature (head/arm), mother_lr, infant_lr indicating which feature we are using in this segment. Besides, there are: mother_init_interaction and infant_init_interaction identifying how many interation initiated by each individual, change_of_leaders indicating the number of changing leaders, and finally correlation_intensity indicating the intensity of the correlation (variance of peak_lags).

Then you can run the following command:
```
python run_analysis_parameters.py
```
You will get a csv file that collects for each video data (of the corresponding dataset) the synchrony parameters. 

The definitions of the parameters included in the normalized_results.csv are the following:

- Average Normalized Mother Initiations = the average among the N data points for each dataset of the normalized number of mother initiations (ratio between # mother initiation / length of the interaction).
- Average Normalized Infant Initiations = the average among the N data points for each dataset of the normalized number of infant initiations (ratio between # infant initiation / length of the interaction).
- Average Normalized Change of Leading  = the average among the N data points for each dataset of the normalized number of leading changes (ratio between # leading changes / length of the interaction). The leading changes is the the number of times that leading has been switched between mother and infants.
- Average Normalized Correlation Intensity = the average among the N data points for each dataset of the normalized correlation intensity (ratio between mean lag variance  / length of the interaction). The correlation intensity is defined as the lags variance

where N is the number of data points of each dataset.

# Contributors
Yiming Luo, Micol Spitale


