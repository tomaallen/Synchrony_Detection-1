# %%
import os
import json
from functions import filter_dict_second, filter_conf_second, read_NaNs
import pandas as pd

from pathlib import Path
import sys
sys.path.append(str(Path(os.getcwd()).parent))
import settings
import data_quality_check as dqc

# %%
# def main(fps: str):
fps = '30' 

# call dqc.get_best_cam
# iterate analysis over best cameras only


filename = "D:\\test\pose_output\PPT0001_2a\json_files\PPT0001_2a-PD-combined_output.json"
f = open(filename)

dict_initial = json.load(f) # initial dictionary

print(len(dict_initial))
total_discarded_frames = 0

########### Initialise Excel file ############
MdRQA_input = os.path.splitext(os.path.basename(filename))[0] + "_NeckNose.csv" 

# worksheet.write(row, 0, "X_Neck_b")
# worksheet.write(row, 1, "Y_Neck_b")
# worksheet.write(row, 2, "X_Nose_b")
# worksheet.write(row, 3, "Y_Nose_b")
# worksheet.write(row, 4, "X_Neck_m")
# worksheet.write(row, 5, "Y_Neck_m")
# worksheet.write(row, 6, "X_Nose_m")
# worksheet.write(row, 7, "Y_Nose_m")


########### Re-arrange initial data ###########
# From a dict of dict to lists containing n=fps dicts
one_second = [] # list that contains all frames in 1 second window
all_seconds = [] # nested list that contains all one second windows

for frame, info in dict_initial.items():
    #print(frame)
    #print(info)
    if (int(frame) % int(fps) == 0) & (int(frame) != 0): # if frame is a multiple of fps
        d1 = {frame : info}
        one_second.append(d1)
        all_seconds.append(one_second)
        one_second = []

    else: # if frame it is NOT a multiple 
        d1 = {frame : info}
        one_second.append(d1)

f.close() # close file

# main('30')
########### Analyse the data ###########
for i in range(len(all_seconds)):
    nans = 0 # NaN will identificate a second with no good frames 
    discarded_frames_second = 0 # total number of discarded frames in 1 second
    for j in range(len(all_seconds[i])):
        seconds_dict_filtered, discarded_frames_temp, row = filter_dict_second(all_seconds[i][j], worksheet, row) # filters dict by discarding frames with less than 2 ppl and frames with no baby-mom couple detected
        seconds_dict_final, discarded_frames, row = filter_conf_second(seconds_dict_filtered, discarded_frames_temp, worksheet, row) # filters dict by discarding neck and hip keypoint with conf level <.3
        
        if(discarded_frames == 1):
            discarded_frames_second += 1
    print("\nThe number of discarded frames in this second is: " + str(discarded_frames_second))
    total_discarded_frames = total_discarded_frames + discarded_frames_second 



print()
print('Intial dict:')
print(len(dict_initial)) 
print('Total Discarded frames:')
print(total_discarded_frames)
print('Total Good Frames:')
print(len(dict_initial)-total_discarded_frames)


########## Read Excel NaNs ################
#read_NaNs(workbook_name)

if __name__ == '__main__':
    
        # Create the parser
    my_parser = argparse.ArgumentParser(description='Process some arguments')
    # Add the arguments
    my_parser.add_argument('fps',
                            type=str,
                            help='video fps, same for all videos')

    # Execute the parse_args() method
    args = my_parser.parse_args()

    main(args.fps)