import shutil
import os
import argparse
import csv
import subprocess, shlex
from functions import filter_dict, filter_conf
import json

if __name__ == '__main__':

    # Create the parser
    my_parser = argparse.ArgumentParser(description='Process some arguments')
    # Add the arguments
    my_parser.add_argument('--folder',
                           type=str,
                           help='the folder containing both face_detect_output and pose_detect_output')

    # Execute the parse_args() method
    args = my_parser.parse_args()
    
    # set paths within args.folder
    face_path = os.path.join(args.folder, 'face_detect_output')
    pose_path = os.path.join(args.folder, 'pose_detect_output')
    
    # reset info file
    analysis_info = os.path.join(args.folder, "analysis_info", "analysis_info.csv")
    if not os.path.isdir(os.path.join(args.folder, "analysis_info")): # make analysis info folder if does not exist
        os.mkdir(os.path.join(args.folder, "analysis_info"))
    with open(analysis_info, 'w', newline="") as f:
        write = csv.writer(f)
        write.writerows([['Filename','Face_attempted','Face_complete','Pose_attempted',
                          'Pose_complete','Csv_copied','Json_combined',
                          'Plots_run','Parameters_run']])

    # create data quality check file
    data_quality = os.path.join(args.folder, "analysis_info", "data_quality.csv")
    with open(data_quality, 'w', newline="") as f:
        write = csv.writer(f)
        write.writerows([['Filename', 'Total frames', 'Discarded frames', 'Good frames']])
    
    # get info on analysis, copy face csv to pose folder and create combined json
    for file in os.listdir(args.folder):
        
        # initialise info variables
        face_attempted = face_completed = face_runs_check = pose_attempted \
            = pose_completed = csv_copied = json_combined = False
        
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

                # data quality check
                f = open(combined_json_path)
                dict_initial = json.load(f) # initial dictionary
                dict_filtered, discarded_frames_temp = filter_dict(dict_initial) # filters dict by discarding frames with less than 2 ppl and frames with no baby-mom couple detected
                dict_final, discarded_frames = filter_conf(dict_filtered, discarded_frames_temp) # filters dict by discarding neck and hip keypoint with conf level <.3
                print('\n # of Discarded frames: ' + str(discarded_frames) + '\n')   
                print('The length of the initial dictionary is ' + str(len(dict_initial)) + '\n')
                print('The length of the final dictionary is ' + str(len(dict_final)))

                # Fill csv
                with open(data_quality, 'a', newline="") as f:
                    write = csv.writer(f)
                    write.writerows([[os.path.basename(combined_json_path).removesuffix("-PD-combined_output.json"),
                                    len(dict_initial),
                                    discarded_frames,
                                    len(dict_final)]])
                                    

            # write analysis info to csv file in args.folder
            with open(analysis_info, 'a', newline="") as f:
                write = csv.writer(f)
                write.writerows([[vid, face_attempted, face_completed, pose_attempted, 
                                  pose_completed, csv_copied, json_combined]])
