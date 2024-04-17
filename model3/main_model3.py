import numpy as np
from sklearn.metrics import mutual_info_score
import os
import json
import xlsxwriter
from pathlib import Path
import numpy as np
import igraph as ig
import statistics
from functions import *



if __name__ == "__main__":
    
    # rootdir =r'folder path containing the JSON combined files' # Edit
    # base_path = r'folder path where you want to save the analysis data' # Edit

    rootdir =r'C:\Users\isabellasole.bisio\Desktop\PCIAnalysis_AllModels\BRAINRISE_datasets\Jsons (from Tom)\complete_dataset\timepoint3-test' # Edit
    base_path = r'C:\Users\isabellasole.bisio\Desktop\PCIAnalysis_AllModels\model3\upload' # Edit
    fps = 30 # Edit
    
    workbook_name = 'EpochsCheck.xlsx'
    workbook = xlsxwriter.Workbook(os.path.join(base_path, workbook_name)) 
    worksheet = workbook.add_worksheet("EpochsCheck")  
    graph_parameters_babyMom = base_path + r"\GraphMetrics_baby-mom.txt"   
    graph_parameters_momBaby = base_path + r"\GraphMetrics_mom-baby.txt" 


    # Create folders
    velocity_vectors_baby = create_folder(base_path, 'baby_velocities')
    velocity_vectors_mom = create_folder(base_path, 'mom_velocities')
    transfer_entropy_babyMom = create_folder(base_path, 'baby-mom_te')
    transfer_entropy_momBaby = create_folder(base_path, 'mom-baby_te')
    subsets_babyMom = create_folder(base_path, 'baby-mom_subsets')
    subsets_momBaby = create_folder(base_path, 'mom-baby_subsets')
    pvalues_babyMom = create_folder(base_path, 'baby-mom_pvalues')
    pvalues_momBaby = create_folder(base_path, 'mom-baby_pvalues')

    # Keypoints to build the weighted-directional graph
    keypoints = ["nose_b", "neck_b", "r_wrist_b", "r_elbow_b", "l_wrist_b", "l_elbow_b", 
                "nose_m", "neck_m", "r_wrist_m", "r_elbow_m", "l_wrist_m", "l_elbow_m"]



    # 1: Frames Check and Velocity Vectors
    row = 0
    worksheet.write(row, 0, "Participant")
    worksheet.write(row, 1, "Total Frames")
    worksheet.write(row, 2, "Discarded Frames")
    worksheet.write(row, 3, "Good Frames")
    worksheet.write(row, 4, "Total Epochs")
    worksheet.write(row, 5, "Good Epochs 3 sec") 
    worksheet.write(row, 6, "Good Epochs 5 sec") 




    for subdir, dirs, files in os.walk(rootdir):
        words = subdir.split('\\')
        if(len(files)!=0):
            for k in range(len(files)):
                print('######### Processing file: ' + files[k] + ' ######################\n')
        
                f = open(rootdir + '\\' + words[len(words)-1] + '\\' + files[k])

                dict_initial = json.load(f) # initial dictionary
                discarded_frames, good_epochs_3sec, good_epochs_5sec, num_tot_epochs, total_good_epochs = check_epochs(dict_initial, fps) # filters dict by discarding frames with less than 2 ppl, frames with no baby-mom couple detected, keypoints with conf score <.3
                print('\n # of Discarded frames: ' + str(discarded_frames) + '\n')   

                # Fill excel spreadsheet 
                row += 1 # new row of the excel file
                worksheet.write(row, 0, files[k])
                worksheet.write(row, 1, len(dict_initial))
                worksheet.write(row, 2, discarded_frames)
                worksheet.write(row, 3, len(dict_initial)-discarded_frames)
                worksheet.write(row, 4,num_tot_epochs)
                worksheet.write(row, 5, good_epochs_3sec)
                worksheet.write(row, 6, good_epochs_5sec)

            
                
                baby_epochs, mom_epochs = evaluate_mom_baby_epochs(total_good_epochs)

                # Evaluate velocities 
                mom_velocities = process_data(mom_epochs)
                baby_velocities = process_data(baby_epochs)


                save_to_file(baby_velocities, velocity_vectors_baby, files[k]+".txt") 
                save_to_file(mom_velocities, velocity_vectors_mom, files[k]+".txt") 

                # Closing file
                f.close()

                        
    workbook.close()
    

    # 2: Transfer Entropy and p-Values
    try:
        # Retrieve the adjacency matrices with Transfer Entropy method
        # (for statistic: it creates baby1-mom1, baby1-mom2, ..., baby1-momN, ..., babyN-momN matrices) 
        start_analysis_random(velocity_vectors_baby, velocity_vectors_mom, transfer_entropy_babyMom)
        start_analysis_random(velocity_vectors_mom, velocity_vectors_baby, transfer_entropy_momBaby)

        # Produce the permutation subset   
        create_subfolder(transfer_entropy_babyMom, subsets_babyMom)
        create_subfolder(transfer_entropy_momBaby, subsets_momBaby)


        # Perform the significance check by creating p-values matrices for each permutation sets (only 3 epochs per permutation)
        process_folder(subsets_babyMom, pvalues_babyMom)
        process_folder(subsets_momBaby, pvalues_momBaby)

    except:
        print("The folder has not been processed")


    # 3: Graph Parameters
    compute_indices(pvalues_babyMom, keypoints, graph_parameters_babyMom)   
    compute_indices(pvalues_momBaby, keypoints, graph_parameters_momBaby)  


    
           
                
            