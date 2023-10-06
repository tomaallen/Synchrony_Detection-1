import shlex
import subprocess
import os
import argparse
import shutil

# run with cd 1_FaceDetection
if __name__ == '__main__':

    # Create the parser
    my_parser = argparse.ArgumentParser(description='Process some arguments')
    # Add the arguments
    my_parser.add_argument('--folder',
                           type=str,
                           help='the folder containing both face_detect_output and pose_detect_output',
                           default=r'')
    my_parser.add_argument('--model_path',
                           type=str,
                           help='path of detect_face.py',
                           default=r'C:\\Users\\bllca\\Synchrony_Detection_streamline\\1_FaceDetection\\detect_face.py')
    
    # TODO: add video fps as an argument here

    # Execute the parse_args() method
    opt = my_parser.parse_args()
    
    PATH = opt.folder # path to trimmed videos
    save_path = os.path.join(PATH, 'face_detect_output\\') # 'D:\\BRAINRISE_PCI\\face_detect_output'
    model_path = opt.model_path
    
    # make face detection output folder within main video folder
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
        
    # clear runs directory of previous unfinished analysis
    if os.path.exists(os.path.join(os.path.dirname(opt.model_path), 'runs')):
        shutil.rmtree(os.path.join(os.path.dirname(opt.model_path), 'runs'))
        print(os.path.join(os.path.dirname(opt.model_path), 'runs') + ' cleared')
    
    # for sub_path in os.listdir(PATH):
    #     if sub_path.endswith('.mp4'):
    #         sub_path_no_ext = os.path.splitext(sub_path)[0]
    #         new_save_path = os.path.join(save_path, sub_path_no_ext)
    #         # print(new_save_path)
    #         if not os.path.isdir(new_save_path):
    #             os.mkdir(new_save_path)
    #             p_vid = os.path.join(PATH, sub_path)
    #             command = 'python '+ model_path+ " " + '-input_path '+p_vid+' '+'-txt_output_directory '+new_save_path+' '+'-yolo_exp_name '+sub_path_no_ext
    #             args = shlex.split(command, posix = 0)
    #             print(args)
    #             # print(command)
    #             subprocess.call(args)
