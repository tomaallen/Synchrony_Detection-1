import shutil
import os
import argparse
import csv
import subprocess, shlex

if __name__ == '__main__':

    # Create the parser
    my_parser = argparse.ArgumentParser(description='Process some arguments')
    # Add the arguments
    my_parser.add_argument('--folder',
                           type=str,
                           help='the folder containing both face_detect_output and pose_detect_output')
    my_parser.add_argument('--fps',
                           type=str,
                           help='video fps')

    # Execute the parse_args() method
    args = my_parser.parse_args()
    
    # set paths within args.folder
    face_path = os.path.join(args.folder, 'face_detect_output')
    pose_path = os.path.join(args.folder, 'pose_detect_output')
    cross_corr_path = os.path.join(args.folder, 'cross_corr_output_default')
    if not os.path.isdir(cross_corr_path): # make cross_corr output folder if does not exist
        os.mkdir(cross_corr_path)
    
    # reset info file
    analysis_info = os.path.join(args.folder, "analysis_info", "analysis_info.csv")
    if not os.path.isdir(os.path.join(args.folder, "analysis_info")): # make analysis info folder if does not exist
        os.mkdir(os.path.join(args.folder, "analysis_info"))
    with open(analysis_info, 'w', newline="") as f:
        write = csv.writer(f)
        write.writerows([['Video','Face_attempted','Face_complete','Pose_attempted',
                          'Pose_complete','Csv_copied','Json_combined',
                          'Plots_run','Parameters_run']])
        
    # reset combined_results file and initialise list of results to add
    combined_results = os.path.join(cross_corr_path, "combined_results.csv")
    with open(combined_results, 'w', newline="") as f:
        write = csv.writer(f)
        write.writerows([['Participant', 'mother_initiated_interactions',
                          'infant_initiated_interactions', 'change_of_leaders', 
                          'intensity(variance)']])
    results_to_add = []

    
    # get info on analysis, copy face csv to pose folder and create combined json
    for file in os.listdir(args.folder):
        
        # initialise info variables
        face_attempted = face_completed = face_runs_check = pose_attempted \
            = pose_completed = csv_copied = json_combined = plots_run \
                = parameters_run = False
        
        if file.endswith('.mp4'): # find all videos in args.folder
        
            # variables for copying face csv to pose folder
            vid = os.path.splitext(file)[0] # remove mp4 extension
            face_csv_file = os.path.join(face_path, vid, vid + '.csv')
            pose_facecsv_folder = os.path.join(pose_path, vid, 'csv_face')
            target_file = os.path.join(pose_facecsv_folder, vid + '.csv')
            
            # path for combined json file
            combined_json_path = os.path.join(pose_path, vid, 'json_files', vid + '-PD-combined_output.json')

            # check if face and pose detection have run
            face_attempted = os.path.isdir(os.path.join(face_path, vid))
            face_completed = face_attempted and os.path.exists(face_csv_file) # face completed if csv created
            pose_attempted = os.path.isdir(os.path.join(pose_path, vid))
            pose_completed = pose_attempted and \
                os.path.exists(os.path.join(pose_path, vid, 'json_files', 
                                            vid + '-PD-output.json')) # pose completed if json created
            
            
            # run combined analysis if face and pose completed
            if face_completed and pose_completed:
                
                # copy face csv to pose folder
                if not os.path.isdir(pose_facecsv_folder): 
                    os.mkdir(pose_facecsv_folder)
                    # print('pose facecsv folder was created: ' + pose_facecsv_folder)
                if not os.path.exists(target_file):
                    shutil.copyfile(face_csv_file, target_file) # copy the face csv file
                    # print(vid + ': csv copied  to ' + target_file)
                    
                csv_copied = os.path.exists(target_file)
                
                
                # create combined json
                if not os.path.exists(combined_json_path):
                    command = 'python create_combined_json.py --reach_dir ' \
                        + os.path.join(pose_path, vid)
                    print(shlex.split(command, posix = 0))
                    subprocess.check_call(shlex.split(command, posix = 0))
                
                json_combined = os.path.exists(combined_json_path)
                
                
                # run analysis plots and parameters
                if not os.path.exists(os.path.join(cross_corr_path, vid, vid + "-segmented_analysis.json")):
                    command = 'python run_analysis_plots.py --PCI_dir ' + cross_corr_path \
                        + ' --reach_dir ' + os.path.join(pose_path, vid) \
                            + ' --plot_on False' \
                                + ' --fps ' + str(args.fps) # no default anymore
                    print(shlex.split(command, posix = 0))
                    subprocess.check_call(shlex.split(command, posix = 0))
                
                plots_run = os.path.exists(os.path.join(cross_corr_path, vid, vid + "-segmented_analysis.json"))
                
                if plots_run: # plots must run first
                    if not os.path.exists(os.path.join(cross_corr_path, vid, vid + "-normalized_results.csv")):
                        command = 'python run_analysis_parameters.py --PCI_dir ' \
                            + cross_corr_path + ' --reach_dir ' + os.path.join(pose_path, vid)
                        print(shlex.split(command, posix = 0))
                        subprocess.check_call(shlex.split(command, posix = 0))
                
                parameters_run = os.path.exists(os.path.join(cross_corr_path, vid, vid + "-normalized_results.csv"))
                
                
                # add parameters to list of results
                if parameters_run:
                    with open(os.path.join(cross_corr_path, vid, vid + "-normalized_results.csv"), newline='') as csvfile:
                        reader = csv.reader(csvfile, delimiter=',')
                        for idx, row in enumerate(reader):
                            if idx == 1:
                                results_to_add.append(row)
            
            
            # write analysis info to csv file in args.folder
            with open(analysis_info, 'a', newline="") as f:
                write = csv.writer(f)
                write.writerows([[vid, face_attempted, face_completed, pose_attempted, 
                                  pose_completed, csv_copied, json_combined, 
                                  plots_run, parameters_run]])
               
                
    # combine csv
    with open(combined_results, 'a', newline="") as f:
        write = csv.writer(f)
        write.writerows(results_to_add)
