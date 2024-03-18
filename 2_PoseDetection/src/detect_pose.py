#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os.path
import sys
# sys.path.remove('C:\\Users\\PC\\AppData\\Roaming\\Python\\Python37\\site-packages')
# print('After remove', sys.path)
from Input import Input
from Scene import Scene

import Constants
import pygame
from os import listdir
from os.path import isfile, join
import reaching_detection._0_data_constants as reaching_const
import argparse
from utils import isMediaFile
from create_json_files import create_json

from pathlib import Path
parent_dir = str(Path(os.getcwd()).parents[1])
sys.path.append(parent_dir)
import settings


class Twister():

    def __init__(self, file_name):
        # print('----------------',Constants.SCREEN_WIDTH, Constants.SCREEN_HEIGHT)
        self.input = Input(file_name=file_name)
        # print('------------------', no_view)
        if not no_view:
            pygame.init()
            # print('----------------', Constants.SCREEN_WIDTH, Constants.SCREEN_HEIGHT)
            pygame.display.set_mode((Constants.SCREEN_WIDTH, Constants.SCREEN_HEIGHT))
            pygame.display.set_caption("Twister!")
            screen = pygame.display.get_surface()
            self.scene = Scene(screen, self.input)

    def run(self):
        no_frames = 0
        while True:
            if self.input.run(no_frames) == 0:
                break
            # print('------------------', no_view)
            if not no_view:
                self.scene.run()
            print("End of frame number:", no_frames)
            # print("==========================================================")
            no_frames += 1

        create_json(os.path.normpath(self.input.folder_csv))


if __name__ == "__main__":

    # Create the parser
    my_parser = argparse.ArgumentParser(description='Process some arguments')
    # Add the arguments
    my_parser.add_argument('--input',
                           type=str,
                           help='the file(s) or folder of input images and/or videos',
                           default=None)
    my_parser.add_argument('--output',
                           type=str,
                           help='the folder of output images and/or videos',
                           default=reaching_const.OUTPUT_FOLDER)
    my_parser.add_argument('--noview',
                           action='store_true',
                           help='noview is True meaning no output videos are displayed during processing')
    my_parser.add_argument('--trackid',
                           action='store_true',
                           help='trackid is True meaning save track id csv files')
    my_parser.add_argument('--skeleton',
                           action='store_true',
                           help='skeleton is True meaning skeleton results')
    my_parser.add_argument('--noheader',
                           action='store_true',
                           help='noheader is True meaning no header for csv files')
    my_parser.add_argument('--outtype',
                           type=str,
                           help='the file(s) or folder of input images and/or videos',
                           default=reaching_const.OUTPUT_TYPE)

    # Execute the parse_args() method
    args = my_parser.parse_args()

    no_view = args.noview
    reaching_const.OUTPUT_FOLDER = args.output
    reaching_const.TRACK_ID = args.trackid
    reaching_const.SKELETON = args.skeleton
    reaching_const.NO_HEADER = args.noheader
    reaching_const.OUTPUT_TYPE = args.outtype

    # to run on individual file, pass input argument
    if args.input:
        if os.path.isfile(args.input):
            file_name = os.path.basename(args.input)
            print('***************************************')
            print(file_name)
            print('***************************************')
            reaching_const.INPUT_FOLDER = os.path.dirname(args.input)
            if isMediaFile(file_name) in ['video', 'image']:
                game = Twister(file_name=file_name)
                game.run()
            else:
                print("Not a video or image!")

    else: # to run on flat folder structure, do not pass input argument
        if os.path.isdir(settings.FOLDER):
            print('Looking for subfolders...........')
            for file in os.listdir(settings.FOLDER):
                if file.endswith(".mp4"):
                    print('***************************************')
                    print(file)
                    print('***************************************')
                    reaching_const.INPUT_FOLDER = str(settings.FOLDER)
                    reaching_const.OUTPUT_FOLDER = str(settings.POSE_FOLDER)
                    print(reaching_const.INPUT_FOLDER)
                    print(reaching_const.OUTPUT_FOLDER)

                    # make output directory if it does not exist
                    if not os.path.isdir(reaching_const.OUTPUT_FOLDER):
                        os.mkdir(reaching_const.OUTPUT_FOLDER)

                    if isMediaFile(file) in ['video', 'image']:
                        # print("RUNNING folder")
                        game = Twister(file_name=file)
                        game.run()
                    else:
                        print("Not a video or image!")

        else:
            raise Exception('FOLDER in settings.py is not a directory')
        
