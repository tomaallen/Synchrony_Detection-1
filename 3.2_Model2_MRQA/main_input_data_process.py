import os
import json
from functions import filter_dict_second, filter_conf_second, read_NaNs
import xlsxwriter
import openpyxl 




def main():
    
    ########### Open the JSON file ###########
    rootdir ='Z:\PCI-PD\Brazil_PCI' # path to JSON combined file and read the file 
    for subdir, dirs, files in os.walk(rootdir):
        words = subdir.split('\\')
        if (str(words[len(words)-1]) == 'json_files'): 
            if ((files[len(files)-2]) == (words[len(words)-2] + '-PD-combined_output.json')):
                print('######### Processing file: ' + files[len(files)-2] + ' ######################\n')
                name = files[len(files)-2].split('-PD-combined_output.json')

                f = open(rootdir + '\\' + words[len(words)-2] + '\\' + words[len(words)-1] + '\\' + files[len(files)-2])

                dict_initial = json.load(f) # initial dictionary
    
                print(len(dict_initial))
                total_discarded_frames = 0

                ########### Initialise Excel file ############
                workbook_name = name[0] +'_NeckNose.xlsx' 
                workbook = xlsxwriter.Workbook(workbook_name) # create a new excel sheet
                worksheet = workbook.add_worksheet("Coordinates") # create a new worksheet

                row = 0
                worksheet.write(row, 0, "X_Neck_b")
                worksheet.write(row, 1, "Y_Neck_b")
                worksheet.write(row, 2, "X_Nose_b")
                worksheet.write(row, 3, "Y_Nose_b")
                worksheet.write(row, 4, "X_Neck_m")
                worksheet.write(row, 5, "Y_Neck_m")
                worksheet.write(row, 6, "X_Nose_m")
                worksheet.write(row, 7, "Y_Nose_m")


                ########### Re-arrange initial data ###########
                # From a dict of dict to lists containing 25/30 dict (1 second, depending of the fps used)
                all_seconds = [] # list that contains all seconds
                one_second = [] # list that contains 1 second
                
                for frame, info in dict_initial.items():
                    #print(frame)
                    #print(info)
                    #if (int(frame) % 25 == 0) & (int(frame) != 0): # if frame is a multiple of 25
                    if (int(frame) % 30 == 0) & (int(frame) != 0): # if frame is a multiple of 25
                        d1 = {frame : info}
                        one_second.append(d1)
                        all_seconds.append(one_second)
                        one_second = []

                    else: # if frame it is NOT a multiple 
                        d1 = {frame : info}
                        one_second.append(d1)
                
                f.close() # close file


            
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

            
                workbook.close() 

                
                
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
	main()