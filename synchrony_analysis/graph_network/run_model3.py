import numpy as np
from sklearn.metrics import mutual_info_score
import os
from glob import glob
import json
import xlsxwriter
from pathlib import Path
import numpy as np
import igraph as ig
import statistics
from functions import *
import argparse
import pandas as pd

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import constants
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import settings

def read_graph_net_txt(file_path):
    # Define a list to store nested data
    data_nested = []

    # Open the file and read line by line
    with open(file_path, 'r') as file:
        lines = file.readlines()
        i = 0
        while i < len(lines):
            if lines[i].startswith("Participant:"):
                # Extract participant name
                participant = lines[i].split(": ")[1].strip()

                # Extract densities
                densities = lines[i + 1].split(": ")[1].strip().split(", ")
                densities = [np.nan if d=='NA' else float(d) for d in densities]

                # Extract strengths
                strengths = lines[i + 2].split(": ")[1].strip().split(", ")
                strengths = [np.nan if s=='NA' else float(s) for s in strengths]

                # Append data as nested list
                data_nested.append([participant, densities, strengths])

                # Move to next participant
                i += 3
            else:
                # Move to next line
                i += 1

    # Convert nested list into long format DataFrame
    data_long = []
    for row in data_nested:
        participant = row[0]
        densities = row[1]
        strengths = row[2]
        for epoch, (density, strength) in enumerate(zip(densities, strengths), start=1):
            data_long.append([participant, str(epoch), density, strength])

    # Create DataFrame
    df_long = pd.DataFrame(data_long, columns=['filename', 'epoch', 'Density', 'Strength'])

    return df_long


def main(rootdir, base_path, fps):

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
        
                # f = open(rootdir + '\\' + words[len(words)-1] + '\\' + files[k])
                f = open(rootdir + '\\' + files[k])

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






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process root directory, base path, and fps")
    parser.add_argument("fps", type=int, help="Frames per second")
    parser.add_argument("--rootdir", type=str, default=settings.FOLDER, help="Root directory of data")
    parser.add_argument("--base_path", type=str, default=settings.MODEL3_FOLDER, help="Base path")
    parser.add_argument("--batch_size", type=int, default=90, help="number of json files analysed at a time (permutations)")

    args = parser.parse_args()

    # split all data into batches
    json_files = []
    for root, dirs, files in os.walk(args.rootdir):
        for file in files:
            if file.endswith(constants.JSON_COMBINED_FILE):
                json_files.append(os.path.join(root, file))

    json_batches = [json_files[i:i+args.batch_size] for i in range(0, len(json_files), args.batch_size)]

    # fill last batch to match batch size from penultimate batch - duplicate
    # results for reused videos are discarded at end of pipeline
    if len(json_batches[-1]) < args.batch_size:
        empty_spots = args.batch_size - len(json_batches[-1])
        json_batches[-1] = json_batches[-1] + json_batches[-2][-empty_spots:]

    # copy jsons into batch folder structure and run
    folder_prefix = 'batch'
    for i, batch in enumerate(json_batches):
        input_folder = os.path.join(args.base_path, f"{folder_prefix}_{i+1}", "json_data")
        os.makedirs(input_folder, exist_ok=True)

        output_folder = os.path.join(args.base_path, f"{folder_prefix}_{i+1}")
        os.makedirs(output_folder, exist_ok=True)

        print(f'Copying batch {i+1} to new folder')
        for file in batch:
            new_file = os.path.join(input_folder, os.path.basename(file))
            if file != new_file:
                shutil.copy(file, new_file)
        
        print(f'Analysing batch {i+1}')
        # main(input_folder, output_folder, args.fps)

    # automatically combine results - baby to mum
    print('Combining baby-to-mum results')
    results_path_bm = os.path.join(args.base_path, 'GraphMetrics_baby-mom.txt')

    # clear file contents if exists
    with open(results_path_bm, 'w') as file:
        pass

    # append all results
    for result in glob(args.base_path + "\\*\\GraphMetrics_baby-mom.txt"):
        print(result)
        with open(result, 'r') as file:
            with open(results_path_bm, 'a') as combined_file:
                combined_file.write(file.read())

    # automatically combine results - mum to baby
    print('Combining mum-to-baby results')
    results_path_mb = os.path.join(args.base_path, 'GraphMetrics_mom-baby.txt')

    # clear file contents if exists
    with open(results_path_mb, 'w') as file:
        pass

    # append all results
    for result in glob(args.base_path + "\\*\\GraphMetrics_mom-baby.txt"):
        print(result)
        with open(result, 'r') as file:
            with open(results_path_mb, 'a') as combined_file:
                combined_file.write(file.read())

    # TODO: read as csv instead and remove duplicates created by last batch dupes
    # %% read graph network data
    graph_net_bm = read_graph_net_txt(results_path_bm)
    graph_net_bm.drop_duplicates(subset='filename').to_csv(os.path.splitext(results_path_bm)[0] + ".csv")

    graph_net_mb = read_graph_net_txt(results_path_mb)
    graph_net_mb.drop_duplicates(subset='filename').to_csv(os.path.splitext(results_path_mb)[0] + ".csv")


           
                
            