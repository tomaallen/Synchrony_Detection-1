import shutil
import os
import argparse

if __name__ == '__main__':

    # Create the parser
    my_parser = argparse.ArgumentParser(description='Process some arguments')
    # Add the arguments
    my_parser.add_argument('--input',
                           type=str,
                           help='the face detection output folder',
                           default=r'Y:\Synchronised_trimmed_videos_STT\Face_detect_output\Mom_cam')
    my_parser.add_argument('--output',
                           type=str,
                           help='the reaching detection output folder',
                           default=r'Y:\Synchronised_trimmed_videos_STT\Pose_detect_output\Mom_Cam')

    # Execute the parse_args() method
    args = my_parser.parse_args()

    for ppt in os.listdir(args.input):
        orig_file = os.path.join(args.input, ppt, ppt + ".csv")
        print(ppt)

        # check if csv face folder exists, create if not
        csv_face_folder = os.path.join(args.output, 'mom_cam'+ ppt + '_adult_synced_trimmed', 'csv_face')
        if not os.path.isdir(csv_face_folder):
            os.mkdir(csv_face_folder)
            print('csv face folder was created for ' + ppt)

        # copy csv face file if no csv file of correct name exists yet
        target_file = os.path.join(csv_face_folder, ppt + '.csv')
        
        if not os.path.exists(target_file):
            shutil.copyfile(orig_file, target_file)
            print('csv copied  for ' + ppt)
