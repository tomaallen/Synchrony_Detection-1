import openpyxl 
import os
from scipy.spatial.distance import euclidean
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import mutual_info_score
import matplotlib.pyplot as plt
import numpy as np
import pyinform
from scipy.stats import entropy
import random
import shutil
import igraph as ig
import statistics


def read_txt_file(file_path):
    with open(file_path, 'r') as file:
        content = json.load(file)
    return content



def start_analysis_random(folder1_path, folder2_path, transfer_e_path): 
    folder1 = Path(folder1_path)
    folder2 = Path(folder2_path)

    all_baby_content = []
    all_mom_content = []
    all_file_names_baby = []
    all_file_names_mom = []

    for file_name in os.listdir(folder1):
        if file_name.endswith('.txt'):
            baby_file_path = folder1 / file_name
            mom_file_path = folder2 / file_name

            if mom_file_path.is_file():
                baby_content = read_txt_file(baby_file_path)
                mom_content = read_txt_file(mom_file_path)
                all_file_names_baby.append(file_name)
                all_file_names_mom.append(file_name)
                print(f"Processed pair: {baby_file_path} and {mom_file_path}")

                all_baby_content.append(baby_content)
                all_mom_content.append(mom_content)

    # Build a vector containing pairs of each element of baby_vel with each element of mom_vel, and same for the files names 
    all_vel_pairs = [(vel_baby, vel_mom) for vel_baby in all_baby_content for vel_mom in all_mom_content]
 
    names_paired = [(name_baby, name_mom) for name_baby in all_file_names_baby for name_mom in all_file_names_mom]
 

    for i in range(len(all_vel_pairs)):
        # Extract and analyse baby and mom keypoints velocities
        try:
            analyze_velocities(all_vel_pairs[i][0], all_vel_pairs[i][1], names_paired[i][0] + " & " + names_paired[i][1], transfer_e_path) 
        except:
            continue

        

def write_matrices_to_file(file_path, matrices):
    try:
        with open(file_path, 'w') as file:
            for matrix in matrices:
                for row in matrix:
                    file.write("\t".join(map(str, row)) + "\n")
                file.write("\n")  # Add an empty line between matrices
        
    except Exception as e:
        print(f"Error: {e}")



def equalize_lists(list1, list2):
    len1, len2 = len(list1), len(list2)

    if len1 == len2:
        # print("Both lists are already of the same length.")
        return list1, list2
    elif len1 > len2:
        list1 = list1[:len2]
        # print("Trimmed the first list to match the length of the second list.")
    else:
        list2 = list2[:len1]
        # print("Trimmed the second list to match the length of the first list.")

    return list1, list2



def analyze_velocities(baby_velocities, mom_velocities, file_name, transfer_e_path): 

    if not baby_velocities or not mom_velocities:
        print("\n Both baby_velocities and mom_velocities lists must not be empty.")
        print("Participant # ", file_name, " is empty. \n")
        return 

    te_list_of_matrices = [] # this contains one matrix for each epoch 

    max_epochs = min(len(baby_velocities), len(mom_velocities))
    if max_epochs == 0:
        print("Participant # ", file_name, " is empty. \n")
    for epoch_num in range(1, max_epochs + 1):
        baby_epoch_data = baby_velocities[epoch_num - 1]
        mom_epoch_data = mom_velocities[epoch_num - 1]

        if not baby_epoch_data or not mom_epoch_data:
            print(f"Epoch {epoch_num}: One or both lists are empty.")
            continue

        # Extract keypoints velocities from the list of dictionaries
        nose_baby = baby_epoch_data.get(f'epoch {epoch_num}', [{}])[0].get('Nose', [])
        neck_baby = baby_epoch_data.get(f'epoch {epoch_num}', [{}])[1].get('Neck', [])
        relbow_baby = baby_epoch_data.get(f'epoch {epoch_num}', [{}])[2].get('RElbow', [])
        rwrist_baby = baby_epoch_data.get(f'epoch {epoch_num}', [{}])[3].get('RWrist', [])
        lelbow_baby = baby_epoch_data.get(f'epoch {epoch_num}', [{}])[4].get('LElbow', [])
        lwrist_baby = baby_epoch_data.get(f'epoch {epoch_num}', [{}])[5].get('LWrist', [])

        nose_mom = mom_epoch_data.get(f'epoch {epoch_num}', [{}])[0].get('Nose', [])
        neck_mom = mom_epoch_data.get(f'epoch {epoch_num}', [{}])[1].get('Neck', [])
        relbow_mom = mom_epoch_data.get(f'epoch {epoch_num}', [{}])[2].get('RElbow', [])
        rwrist_mom = mom_epoch_data.get(f'epoch {epoch_num}', [{}])[3].get('RWrist', [])
        lelbow_mom = mom_epoch_data.get(f'epoch {epoch_num}', [{}])[4].get('LElbow', [])
        lwrist_mom = mom_epoch_data.get(f'epoch {epoch_num}', [{}])[5].get('LWrist', [])


        nose_baby, nose_mom = equalize_lists(nose_baby, nose_mom)
        neck_baby, neck_mom = equalize_lists(neck_baby, neck_mom)
        relbow_baby, relbow_mom = equalize_lists(relbow_baby, relbow_mom)
        rwrist_baby, rwrist_mom = equalize_lists(rwrist_baby, rwrist_mom)
        lelbow_baby, lelbow_mom = equalize_lists(lelbow_baby, lelbow_mom)
        lwrist_baby, lwrist_mom = equalize_lists(lwrist_baby, lwrist_mom)


        # Transfer Entropy:
        k = 1  # time lag
        if len(nose_baby) == 0: # if the participant has not usable data: len of all the previously returned vectors is 0
            print("\n" + file_name + " has no usable data. \n")
        # IF list is not empty
        else:
            # print("\n Evaluating Transfer Entropy matrices for participant " + file_name + " for epoch # " + str(epoch_num) + "\n" )
            te_noseB_noseM = pyinform.transferentropy.transfer_entropy(nose_baby, nose_mom, k=k)
            te_noseB_neckM = pyinform.transferentropy.transfer_entropy(nose_baby, neck_mom, k=k)
            te_noseB_rwristM = pyinform.transferentropy.transfer_entropy(nose_baby, rwrist_mom, k=k)
            te_noseB_relbowM = pyinform.transferentropy.transfer_entropy(nose_baby, relbow_mom, k=k)
            te_noseB_lwristM = pyinform.transferentropy.transfer_entropy(nose_baby, lwrist_mom, k=k)
            te_noseB_lelbowM = pyinform.transferentropy.transfer_entropy(nose_baby, lelbow_mom, k=k)

            te_neckB_noseM = pyinform.transferentropy.transfer_entropy(neck_baby, nose_mom, k=k)
            te_neckB_neckM = pyinform.transferentropy.transfer_entropy(neck_baby, neck_mom, k=k)
            te_neckB_rwristM = pyinform.transferentropy.transfer_entropy(neck_baby, rwrist_mom, k=k)
            te_neckB_relbowM = pyinform.transferentropy.transfer_entropy(neck_baby, relbow_mom, k=k)
            te_neckB_lwristM = pyinform.transferentropy.transfer_entropy(neck_baby, lwrist_mom, k=k)
            te_neckB_lelbowM = pyinform.transferentropy.transfer_entropy(neck_baby, lelbow_mom, k=k)

            te_rwristB_noseM = pyinform.transferentropy.transfer_entropy(rwrist_baby, nose_mom, k=k)
            te_rwristB_neckM = pyinform.transferentropy.transfer_entropy(rwrist_baby, neck_mom, k=k)
            te_rwristB_rwristM = pyinform.transferentropy.transfer_entropy(rwrist_baby, rwrist_mom, k=k)
            te_rwristB_relbowM = pyinform.transferentropy.transfer_entropy(rwrist_baby, relbow_mom, k=k)
            te_rwristB_lwristM = pyinform.transferentropy.transfer_entropy(rwrist_baby, lwrist_mom, k=k)
            te_rwristB_lelbowM = pyinform.transferentropy.transfer_entropy(rwrist_baby, lelbow_mom, k=k)

            te_relbowB_noseM = pyinform.transferentropy.transfer_entropy(relbow_baby, nose_mom, k=k)
            te_relbowB_neckM = pyinform.transferentropy.transfer_entropy(relbow_baby, neck_mom, k=k)
            te_relbowB_rwristM = pyinform.transferentropy.transfer_entropy(relbow_baby, rwrist_mom, k=k)
            te_relbowB_relbowM = pyinform.transferentropy.transfer_entropy(relbow_baby, relbow_mom, k=k)
            te_relbowB_lwristM = pyinform.transferentropy.transfer_entropy(relbow_baby, lwrist_mom, k=k)
            te_relbowB_lelbowM = pyinform.transferentropy.transfer_entropy(relbow_baby, lelbow_mom, k=k)

            te_lwristB_noseM = pyinform.transferentropy.transfer_entropy(lwrist_baby, nose_mom, k=k)
            te_lwristB_neckM = pyinform.transferentropy.transfer_entropy(lwrist_baby, neck_mom, k=k)
            te_lwristB_rwristM = pyinform.transferentropy.transfer_entropy(lwrist_baby, rwrist_mom, k=k)
            te_lwristB_relbowM = pyinform.transferentropy.transfer_entropy(lwrist_baby, relbow_mom, k=k)
            te_lwristB_lwristM = pyinform.transferentropy.transfer_entropy(lwrist_baby, lwrist_mom, k=k)
            te_lwristB_lelbowM = pyinform.transferentropy.transfer_entropy(lwrist_baby, lelbow_mom, k=k)

            te_lelbowB_noseM = pyinform.transferentropy.transfer_entropy(lelbow_baby, nose_mom, k=k)
            te_lelbowB_neckM = pyinform.transferentropy.transfer_entropy(lelbow_baby, neck_mom, k=k)
            te_lelbowB_rwristM = pyinform.transferentropy.transfer_entropy(lelbow_baby, rwrist_mom, k=k)
            te_lelbowB_relbowM = pyinform.transferentropy.transfer_entropy(lelbow_baby, relbow_mom, k=k)
            te_lelbowB_lwristM = pyinform.transferentropy.transfer_entropy(lelbow_baby, lwrist_mom, k=k)
            te_lelbowB_lelbowM = pyinform.transferentropy.transfer_entropy(lelbow_baby, lelbow_mom, k=k)



            # Create a list of the 36 te elements in the specified order
            elements = [te_noseB_noseM, te_noseB_neckM, te_noseB_rwristM, te_noseB_relbowM, te_noseB_lwristM, te_noseB_lelbowM, 
                        te_neckB_noseM, te_neckB_neckM, te_neckB_rwristM, te_neckB_relbowM, te_neckB_lwristM, te_neckB_lelbowM, 
                        te_rwristB_noseM, te_rwristB_neckM, te_rwristB_rwristM, te_rwristB_relbowM, te_rwristB_lwristM, te_rwristB_lelbowM,  
                        te_relbowB_noseM, te_relbowB_neckM, te_relbowB_rwristM, te_relbowB_relbowM, te_relbowB_lwristM, te_relbowB_lelbowM,  
                        te_lwristB_noseM, te_lwristB_neckM, te_lwristB_rwristM, te_lwristB_relbowM, te_lwristB_lwristM, te_lwristB_lelbowM,
                        te_lelbowB_noseM, te_lelbowB_neckM, te_lelbowB_rwristM,  te_lelbowB_relbowM, te_lelbowB_lwristM, te_lelbowB_lelbowM 
                        ]
         
            # Create a 6x6 matrix
            matrix_te = [[0]*6 for _ in range(6)]

            # Fill the matrix with the given elements
            for i in range(6):
                for j in range(6):
                    matrix_te[i][j] = elements[i*6 + j]

            # Print the matrix in a tabular format
            # for row in matrix_te:
                # print("\t".join(map(str, row)))     

            te_list_of_matrices.append(matrix_te) # append a new matrix for each epoch

        # Create a new txt file containing all the matrices for each epoch
        write_matrices_to_file(transfer_e_path + "\\" + str(file_name) + ".txt", te_list_of_matrices)    # EDIT this with the path where you want to save the matrices



#This function check if a frame is good or not and create epochs with consecutive good frames
def check_epochs(initial_dict, fps):
    discarded_frames = 0
    good_epochs_3sec = 0
    good_epochs_5sec = 0
    check_baby_ma_id = 0
    total_epochs = [] # vector containing all the epochs
    total_good_epochs = [] # vector containing all the good epochs
    epoch = [] # one epoch = vector of consecutive good frames
 
    

    # Check dict
    for frame, info in initial_dict.items():
        keypoint_analysis = 0
        for key, values  in info.items():
            
            # Check how many ppl are detected per frame
            if (key=='Count'): 
                if (values >= 2):
                    #print('2 or More People detected -> Potentially a Good Frame, Check More!\n')
                    check_baby_ma_id = 1   # Check for baby_ma_id          

                else:
                    #print('Not enough people detected -> Discard frame! ')
                    discarded_frames = discarded_frames + 1    
                    check_baby_ma_id = 0   # No need to check for baby_ma_id -> frame already discarded
                    total_epochs.append(epoch)
                    epoch = []

            # Check if there is a mom-baby couple in the frame
            else:
                if (check_baby_ma_id == 1):
                    mom_is_here = 0 
                    baby_is_here = 0

                    # iterate in all people's data
                    for i in range(len(values)):
                        try:
                            for key, data  in values[i].items():
                                if (key == 'baby_ma_id'):
                                    if (data == 0): # is mom
                                        mom_is_here += 1
                                        #print('mom_is_here')
                                        #print(mom_is_here)
                                    elif (data == 1): # is baby
                                        baby_is_here += 1  
                                        #print('baby_is_here') 
                                        #print(baby_is_here)


                        except:
                            mom_is_here = 0
                            baby_is_here = 0




                    # check if we have a couple of mom-baby
                    if (mom_is_here == 1 & baby_is_here == 1):
                        #print()
                        #print('This is a mom-baby couple -> Potentially a Good Frame, Check More!\n')
                        keypoint_analysis = 1
                    
                    else:
                        #print()
                        #print('This is NOT a mom-baby couple -> Discard Frame!\n')
                        discarded_frames = discarded_frames + 1    
                        total_epochs.append(epoch)
                        epoch = []

                else:    
                    break    

        if (keypoint_analysis == 1): # next check will be the keypoint confidence score
            d1 = {'Frame': frame, 'Info': values}
            
            for key, values  in d1.items():
                if key == ('Info'):
                    # #print(values)
                    bad_frame = 0 # flag to keep track if one person involved in the frame has conf score <.3
                    for k in range(len(values)): # Iterates 2 times -> 2 ppl per frame
                        if ((values[k]['baby_ma_id'] == 0) or (values[k]['baby_ma_id'] == 1)):
                            if ((values[k]['Body']['Neck'][2] < 0.3) or (values[k]['Body']['Nose'][2] < 0.3) or (values[k]['Body']['RElbow'][2] < 0.3)
                            or (values[k]['Body']['LElbow'][2] < 0.3) or (values[k]['Body']['RWrist'][2] < 0.3) or (values[k]['Body']['LWrist'][2] < 0.3)): # Checks keypoints confidence score
    
                                bad_frame = bad_frame + 1 # the person analysed has conf score <.3

                    #print(i)
                    if (bad_frame == 0): # both ppl have conf scores >0.3 
                        epoch.append(d1) # This is a GOOD frame!

                    else: # one or both ppl have conf score <.3 
                        #print('\nDiscard this frame!')
                        discarded_frames = discarded_frames +1
                        total_epochs.append(epoch)
                        epoch = []

    for l in range(len(total_epochs)):
        if len(total_epochs[l]) >= (fps*3): # at least 3 seconds (30fps)
            good_epochs_3sec +=1
            total_good_epochs.append(total_epochs[l])

        if len(total_epochs[l]) >= (fps*5): # at least 5 seconds (30fps)
            good_epochs_5sec +=1    



    return discarded_frames, good_epochs_3sec, good_epochs_5sec, len(total_epochs), total_good_epochs



# This function evaluates baby and mom selected keypoints velocities 
def evaluate_mom_baby_epochs(total_epochs):

    # Create mom and baby list of epochs
    baby_epochs = []
    mom_epochs = []

    for i in range(len(total_epochs)):
        temp_baby = []
        temp_mom = []
        for j in range(len(total_epochs[i])):
            for entry in total_epochs[i][j].get('Info', []):
                if 'baby_ma_id' in entry:
                    if entry['baby_ma_id'] == 0:
                        #print('mom')
                        #print(entry)
                        temp_mom.append(entry)
                        #print(temp_mom)
                    elif entry['baby_ma_id'] == 1:
                        #print("baby")
                        #print(entry)
                        temp_baby.append(entry)
                        #print(temp_baby)
        baby_epochs.append(temp_baby)
        mom_epochs.append(temp_mom)

    return baby_epochs, mom_epochs



# Evaluate velocities vectors using Euclidean Distance
def calculate_distances(data):
    key_points = ['Nose', 'Neck', 'RElbow', 'RWrist', 'LElbow', 'LWrist']
    distances = {key: [] for key in key_points}

    for i in range(len(data) - 1):
        current_body = data[i]['Body']
        next_body = data[i + 1]['Body']
        
        for key in key_points:
            current_point = current_body[key]
            next_point = next_body[key]
            distance = euclidean_distance(current_point, next_point)
            distances[key].append(distance)
    
    return distances



def process_data(input_data):
    result = []
    non_empty_count = 0
    for epoch, data in enumerate(input_data, start=1):
        if not data:  # Skip empty lists
            continue
        non_empty_count += 1
        epoch_name = f"epoch {non_empty_count}"
        distances = calculate_distances(data)
        epoch_data = [{key: distances[key]} for key in distances]
        result.append({epoch_name: epoch_data})
    return result    



def euclidean_distance(point1, point2):
    return euclidean(point1, point2)



def save_to_file(data, folder_path, file_name):
    file_path = os.path.join(folder_path, file_name)
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=2)
    # print(f"Data saved to {file_path}")           
            
   

def create_subfolder(input_folder, output_folder):
    print(input_folder)
    print(output_folder)
    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            # Extract participant name from the filename
            participant_name = filename.split("-PD-combined_output")[0]

            # Create or use existing subfolder with participant_name
            subfolder_path = os.path.join(output_folder, participant_name)
            os.makedirs(subfolder_path, exist_ok=True)

            # Check and copy files to the created subfolder
            for file_to_copy in os.listdir(input_folder):
                if file_to_copy.endswith(".txt"):
                    file_to_copy_path = os.path.join(input_folder, file_to_copy)

                    # Check conditions for copying the file
                    if (
                        participant_name not in file_to_copy
                        or file_to_copy.count(participant_name) == 2
                    ):                        
                        shutil.copy(file_to_copy_path, subfolder_path)



def evaluate_mean_std(real_m, all_perm_m):
    rows, cols = len(real_m), len(real_m[0])
    mean_perm = np.zeros((rows, cols))
    std_perm = np.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):
            values_at_position = [matrix[i][j] for matrix in all_perm_m]
            mean_perm[i][j] = np.mean(values_at_position)
            std_perm[i][j] = np.std(values_at_position)

    return mean_perm, std_perm



def evaluate_p_values(real_m, mean_perm, std_perm):
    p_values_m = (real_m - mean_perm) / std_perm
    p_values_m[p_values_m < 0] = 0  # Set negative elements to zero
    return p_values_m



def process_folder(input_folder, output_folder):
    for folder_name in os.listdir(input_folder):
        folder_path = os.path.join(input_folder, folder_name)
        print()
        print("Processing folder : " + folder_name)
        all_perm_matrices = []  # List to store all perm_m_i matrices
        i = 0

        if os.path.isdir(folder_path):
            participant_name = folder_name
            txt_files = [file for file in os.listdir(folder_path) if file.endswith('.txt') and participant_name in file]

            if txt_files:
                txt_file_name = txt_files[0]
                txt_file_path = os.path.join(folder_path, txt_file_name)

                # Read the matrix from the text file
                with open(txt_file_path, 'r') as file:
                    real_m = np.loadtxt(file)

                # Check and modify the dimensions of the matrix
                if real_m.shape == (18, 6):
                    modified_m = real_m
                elif real_m.shape == (6, 6):
                    modified_m = np.concatenate([real_m] * 3, axis=0)
                elif real_m.shape == (12, 6):
                    modified_m = np.concatenate([real_m, real_m[:6, :]], axis=0)
                else:
                    modified_m = real_m[:18, :6]

                # Save the modified matrix
                # output_file_path = os.path.join(folder_path, 'modified_matrix.npy')
                # np.save(output_file_path, modified_m)

                print(f"Processed {participant_name}") #: Saved modified matrix to {output_file_path}")
                # print(modified_m)

                # Process other .txt files
                other_txt_files = [file for file in os.listdir(folder_path) if file.endswith('.txt') and participant_name not in file]

                for i, txt_file_name in enumerate(other_txt_files):
                    txt_file_path = os.path.join(folder_path, txt_file_name)

                    # Read the matrix from the text file
                    with open(txt_file_path, 'r') as file:
                        perm_m = np.loadtxt(file)

                    # Check and modify the dimensions of the matrix
                    if perm_m.shape == (18, 6):
                        modified_perm_m = perm_m
                    elif perm_m.shape == (6, 6):
                        modified_perm_m = np.concatenate([perm_m] * 3, axis=0)
                    elif perm_m.shape == (12, 6):
                        modified_perm_m = np.concatenate([perm_m, perm_m[:6, :]], axis=0)
                    else:
                        modified_perm_m = perm_m[:18, :6]

                    # Save the modified matrix with a unique identifier i
                    # output_file_path = os.path.join(folder_path, f'perm_m_{i}.npy')
                    # np.save(output_file_path, modified_perm_m)

                    print(f"Processed {txt_file_name}, from folder {folder_name}") #: Saved modified matrix to {output_file_path}")
                    # print(modified_perm_m)

                    # Append the modified  permutation matrix to the list
                    all_perm_matrices.append(modified_perm_m)


                # Evaluate the mean, std and p_values matrices
                mean_perm, std_perm = evaluate_mean_std(modified_m, all_perm_matrices)
                p_values_m = evaluate_p_values(modified_m, mean_perm, std_perm)

                #print("P values matrix is : ")
                #print(p_values_m)

                # Save the content of the p values matrix in a new .txt file
                save_file_path = os.path.join(output_folder, folder_name + ".txt")
                np.savetxt(save_file_path, p_values_m, fmt='%f', delimiter='\t')


def read_txt_file_graphParam(filepath):
    """
    Reads a .txt file containing a 18x6 matrix and returns it as a numpy array.
    """
    with open(filepath, 'r') as file:
        lines = file.readlines()
        matrix = []
        for line in lines:
            row = [float(x) for x in line.strip().split()]
            matrix.append(row)
        return np.array(matrix)

def split_matrix(matrix):
    """
    Splits a 18x6 matrix into three 6x6 matrices.
    """
    epoch_1 = matrix[:6]
    epoch_2 = matrix[6:12]
    epoch_3 = matrix[12:]
    return epoch_1, epoch_2, epoch_3


def create_graph(directional_matrix, keypoints):
    # Create a directed graph
    g = ig.Graph(directed=True)

    # Add nodes
    g.add_vertices(keypoints)

    # Add edges with weights
    for i, row in enumerate(directional_matrix):
        for j, value in enumerate(row):
            if value != 0:  # Only add edges for non-zero values
                g.add_edge(keypoints[i], keypoints[j], weight=value)

    return g

def evaluate_metrics(g):
    # Calculate density
    density = g.density()

    # Calculate strength : sum of the edge weights adjacent to a node
    strength = g.strength(weights='weight')  # Use weights for strength calculation
    avg_strength = statistics.mean(strength)



    print("Density:", density)
    print("The avg Strength is:", avg_strength)    

    return density, avg_strength




def save_graph_metrics(output_file, filename, densities, avg_strengths):
    # Check if the file exists and is not empty
    try:
        with open(output_file, "r") as file:
            content = file.read()
            if content.strip():  # Check if content is not empty
                with open(output_file, "a") as file_append:
                    file_append.write("\n")  # Add a blank line before appending new content
                    file_append.write("Participant: {}\n".format(filename))
                    file_append.write("Densities epoch1, epoch2, epoch3: {}, {}, {}\n".format(*densities))
                    file_append.write("Strength epoch1, epoch2, epoch3: {}, {}, {}\n".format(*avg_strengths))
                    file_append.write("\n")  # Add a blank line at the end
    except FileNotFoundError:
        # File doesn't exist, create it and write new content
        with open(output_file, "w") as file:
            file.write("Participant: {}\n".format(filename))
            file.write("Densities epoch1, epoch2, epoch3: {}, {}, {}\n".format(*densities))
            file.write("Strength epoch1, epoch2, epoch3: {}, {}, {}\n".format(*avg_strengths))
            file.write("\n")  # Add a blank line at the end
    

def compute_indices(folder_path, keypoints, output_file):
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            densities = []
            avg_strengths = []
            file_path = os.path.join(folder_path, filename)
            matrix = read_txt_file_graphParam(file_path)
            epoch_1, epoch_2, epoch_3 = split_matrix(matrix)
            
            # Save or process the epoch matrices as needed
            # For demonstration, printing them here
            print(filename)
            print()
            print("Epoch 1:")
            print(epoch_1)
            print("Epoch 2:")
            print(epoch_2)
            print("Epoch 3:")
            print(epoch_3)
            print("="*20)



            # Create graph
            g1 = create_graph(epoch_1, keypoints)
            g2 = create_graph(epoch_2, keypoints)
            g3 = create_graph(epoch_3, keypoints)

            # Evaluate metrics
            try:
                density1, avg_strength1 = evaluate_metrics(g1) 
                densities.append(density1)
                avg_strengths.append(avg_strength1)
            except:
                density1 = 'NA'
                avg_strength1 = 'NA'
                print('Density: ', density1)
                print('Strength: ', avg_strength1)
                densities.append(density1)
                avg_strengths.append(avg_strength1)

            try:
                density2, avg_strength2 = evaluate_metrics(g2) 
                densities.append(density2)
                avg_strengths.append(avg_strength2)
            except:
                density2 = 'NA'
                avg_strength2 = 'NA'
                print('Density: ', density2)
                print('Strength: ', avg_strength2)
                densities.append(density2)
                avg_strengths.append(avg_strength2)

            try:
                density3, avg_strength3 = evaluate_metrics(g3) 
                densities.append(density3)
                avg_strengths.append(avg_strength3)
            except: 
                density3 = 'NA'
                avg_strength3 = 'NA' 
                print('Density: ', density3)
                print('Strength: ', avg_strength3)  
                densities.append(density3)
                avg_strengths.append(avg_strength3)    

            save_graph_metrics(output_file, filename, densities, avg_strengths)    


def create_folder(location, folder_name):
    # Join the location path with the folder name to create the full path
    folder_path = os.path.join(location, folder_name)
    
    try:
        # Create the folder
        os.makedirs(folder_path)
    
    except OSError as e:
        print(f"Failed to create folder '{folder_name}' at '{location}': {e}")

    return folder_path



