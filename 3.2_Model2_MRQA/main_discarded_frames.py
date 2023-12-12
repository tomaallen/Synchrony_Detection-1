import json
import os
from functions import filter_dict, filter_conf
import xlsxwriter



def main():
    rootdir ='Z:\PCI-PD\Singapore_PCI_videos_1kD_3' # path to JSON combined file and read the file 
    workbook = xlsxwriter.Workbook('Result_Neck_Nose_Conf_0.3_Singapore_4.xlsx') # create a new excel sheet
    worksheet = workbook.add_worksheet("Singapore_PCI") # create a new worksheet

    row = 0
    worksheet.write(row, 0, "Participant")
    worksheet.write(row, 1, "Total Frames")
    worksheet.write(row, 2, "Discarded Frames")
    worksheet.write(row, 3, "Good Frames")

    for subdir, dirs, files in os.walk(rootdir):
        words = subdir.split('\\')
        print(words)
        print()
        print(words[len(words)-1])
        if (str(words[len(words)-1]) == 'json_files'): 
            if ((files[len(files)-2]) == (words[len(words)-2] + '-PD-combined_output.json')):
                print('######### Processing file: ' + files[len(files)-2] + ' ######################\n')
                print(rootdir + '\\' + words[len(words)-2] + '\\' + words[len(words)-1] + '\\')

                f = open(rootdir + '\\' + words[len(words)-2] + '\\' + words[len(words)-1] + '\\' + files[len(files)-2])

                dict_initial = json.load(f) # initial dictionary
                dict_filtered, discarded_frames_temp = filter_dict(dict_initial) # filters dict by discarding frames with less than 2 ppl and frames with no baby-mom couple detected
                dict_final, discarded_frames = filter_conf(dict_filtered, discarded_frames_temp) # filters dict by discarding neck and hip keypoint with conf level <.3
                print('\n # of Discarded frames: ' + str(discarded_frames) + '\n')   
                print('The length of the initial dictionary is ' + str(len(dict_initial)) + '\n')
                print('The length of the final dictionary is ' + str(len(dict_final)))

                # Fill excel spreadsheet 
                row += 1 # new row of the excel file
                worksheet.write(row, 0, files[len(files)-2])
                worksheet.write(row, 1, len(dict_initial))
                worksheet.write(row, 2, discarded_frames)
                worksheet.write(row, 3, len(dict_final))


                # Closing file
                f.close()

                #create variables for the different configurations 
                #function: check_left_right() 
                #function: check_front_back()
                #function: check_torso_config()
                #function: assign_torso_score
                #function: check_head_config()
                #function: assign_head_score
                #function: assign_final_score
                #normalise wrt number of frames (this includes norm wrt video duration and confidence score/discarded frames)
    workbook.close()

if __name__ == '__main__':
	main()