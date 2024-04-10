import shutil
import os
import argparse
import csv
import json
import numpy as np
import pandas as pd

from pathlib import Path
import sys
sys.path.append(str(Path(os.getcwd()).parent))
import settings
import data_quality_check as dqc
from json2csv import read_json

import matplotlib.pyplot as plt
from model1.preprocess import get_head_angle, get_arm_angle
from model1.corr_functions import *

import time

if __name__ == '__main__':

    start = time.time()

    # Create the parser
    my_parser = argparse.ArgumentParser(description='Process some arguments')
    # Add the arguments
    my_parser.add_argument('fps',
                            type=str,
                            help='video fps, same for all videos')
    my_parser.add_argument('--plot',
                            type=bool,
                            help='create plots or not - otherwise just make segmented analysis json',
                            default=True)

    # Execute the parse_args() method
    args = my_parser.parse_args()
    
    # set analysis folder with fps shown
    settings.MODEL1_FOLDER = Path(str(settings.MODEL1_FOLDER) + "_" + args.fps + "fps")
    if not os.path.isdir(settings.MODEL1_FOLDER): # make cross_corr output folder if does not exist
        os.mkdir(settings.MODEL1_FOLDER)

    # choose the camera with the most good frames for each participant and timepoint
    best_cam_list = dqc.get_best_cams([['LShoulder', 'LElbow'],
                                        ['RShoulder', 'RElbow'],
                                        ['Nose', 'LEar'],
                                        ['Nose', 'REar']])

    # output headers
    res_rows = [["video",
                "mother_initiated_interactions",
                "infant_initiated_interactions",
                "change_of_leaders",
                "synchrony_stability"]]

    for file in best_cam_list:
        # run analysis plots and parameters
        if not os.path.exists(os.path.join(settings.MODEL1_FOLDER, file, file + "-segmented_analysis.json")):

            ## Correlation analysis: Saving plots
            json_path = os.path.join(settings.POSE_FOLDER, file, "json_files")
            participant_output_path = os.path.join(settings.MODEL1_FOLDER, file)
            os.makedirs(participant_output_path, exist_ok = True) # make participant folder in PCI dir if does not exist
            
            for comb_json_path in os.listdir(json_path):
                if comb_json_path.endswith("combined_output.json"):
                    # json segmented analysis
                    comb_json_path = os.path.join(json_path, comb_json_path)
                    mother_tot, infant_tot = read_json(comb_json_path)
                    mother_signals, infant_signals = {}, {}
                    for part, func in [("arm", get_arm_angle), ("head", get_head_angle)]:
                        mother_signal_l, mother_confidence_l, mother_signal_r, mother_confidence_r = func(mother_tot, 0.1, int(round(3 * int(args.fps))))
                        infant_signal_l, infant_confidence_l, infant_signal_r, infant_confidence_r = func(infant_tot, 0.1, int(round(3 * int(args.fps))))
                        mother_signals[part] = [(mother_signal_l, mother_confidence_l), (mother_signal_r, mother_confidence_r)]
                        infant_signals[part] = [(infant_signal_l, infant_confidence_l), (infant_signal_r, infant_confidence_r)]
                    max_len = len(mother_signals["arm"][0][0])
                    reses = analysis_sequence(mother_signals, infant_signals, fps = int(args.fps), min_seg_length=10) # TODO: add args here for gridsearch
                    output_json_path = os.path.join(participant_output_path, file + "-segmented_analysis.json")
                    output_img_path = os.path.join(participant_output_path, file + "-peak_curve.png")
                    json.dump(reses, open(output_json_path, "w"))
                    
                    # plotting
                    if args.plot:
                        fig, ax = plt.subplots(4, 2, sharex = True, sharey = True)
                        fig.text(0.5, 0.04, "Correlation / Peak Lag (seconds)", ha = "center")
                        fig.text(0.04, 0.5, "Step start time (seconds)", va = "center", rotation = "vertical")
                        for i1, part in enumerate(["arm", "head"]):
                            for j, mlr in enumerate(["L", "R"]):
                                for k, ilr in enumerate(["L", "R"]):
                                    plt.subplot(4, 2, i1 * 4 + j * 2 + k + 1)
                                    plt.title(f"{part.capitalize()} correlation mother {mlr} infant {ilr}")
                                    curves, peak_indices, peak_lags, peak_corrs = [], [], [], []
                                    eps = 1e-7
                                    for res in reses:
                                        if res["feature"] == part and res["mother_lr"] == mlr and res["infant_lr"] == ilr:
                                            curves.append((res["peak_indices"][0], res["peak_indices"], res["peak_lags"], res["peak_corrs"]))
                                    for i in range(len(curves)):
                                        if i > 0:
                                            peak_indices.extend([curves[i - 1][1][-1] + eps, curves[i][1][0] - eps])
                                            peak_lags.extend([np.nan, np.nan])
                                            peak_corrs.extend([np.nan, np.nan])
                                        peak_indices.extend(curves[i][1])
                                        peak_lags.extend(curves[i][2])
                                        peak_corrs.extend(curves[i][3])
                                    plt.plot(peak_indices, peak_lags, label = "Lags")
                                    plt.plot(peak_indices, peak_corrs, label = "Corrs")
                                    plt.legend()
                        plt.subplots_adjust(left = 0.082, wspace = 0.05, right = 0.95, top = 0.95)
                        print("Saving: ", output_img_path)
                        # plt.show()
                        plt.savefig(output_img_path)
                        plt.close(fig)
            
                    # run analysis parameters
                    all_peak_lags = []
                    tot_time, tot_mother_init, tot_infant_init, tot_change = 0, 0, 0, 0
                    for res in reses:
                        dlt_time = res["end_time"] - res["start_time"]
                        tot_time += dlt_time
                        tot_mother_init += res["mother_init_interaction"]
                        tot_infant_init += res["infant_init_interaction"]
                        tot_change += res["change_of_leaders"]
                        all_peak_lags.extend(res["peak_lags"])
                    try:
                        res_rows.append([file, tot_mother_init / tot_time, tot_infant_init / tot_time, tot_change / tot_time, np.var(np.array(all_peak_lags))])
                    except:
                        print("Error with analysis parameters for this participant")
                        res_rows.append([file, np.nan, np.nan, np.nan, np.nan])

    # write results
    with open(os.path.join(settings.MODEL1_FOLDER, "combined_results.csv"), "a", newline = "") as f:
        writer = csv.writer(f)
        writer.writerows(res_rows)

    end = time.time()
    print('Runtime: {}sec'.format(round(end - start)))

