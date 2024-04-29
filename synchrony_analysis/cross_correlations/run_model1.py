import os
from pathlib import Path
from glob import glob
import argparse
import csv
import json
import numpy as np
import time
import multiprocessing as mp
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import constants
from data_quality_check import get_best_cams
from json2csv import read_json
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import settings

import matplotlib.pyplot as plt
from model1.preprocess import get_head_angle, get_arm_angle
from model1.corr_functions import *


def model1(file: str, ppt, tp, args):

    # run analysis plots and parameters
    if not os.path.exists(os.path.join(str(settings.MODEL1_FOLDER) + "_" + args.fps + "fps", file, file + "-segmented_analysis.json")): # check if run already

        ## Correlation analysis: Saving plots
        participant_output_path = os.path.join(str(settings.MODEL1_FOLDER) + "_" + args.fps + "fps", file)
        # print(participant_output_path)
        os.makedirs(participant_output_path, exist_ok = True) # make participant folder in PCI dir if does not exist
        
        comb_json_path = settings.POSE_FOLDER / file / "json_files" / str(file + "-" + constants.JSON_COMBINED_FILE)
        print('Running analysis on ' + str(comb_json_path))

        # json segmented analysis
        mother_tot, infant_tot = read_json(comb_json_path)
        mother_signals, infant_signals = {}, {}
        for part, func in [("arm", get_arm_angle), ("head", get_head_angle)]:
            mother_signal_l, mother_confidence_l, mother_signal_r, mother_confidence_r = func(mother_tot, 0.1, int(round(3 * int(args.fps))))
            infant_signal_l, infant_confidence_l, infant_signal_r, infant_confidence_r = func(infant_tot, 0.1, int(round(3 * int(args.fps))))
            mother_signals[part] = [(mother_signal_l, mother_confidence_l), (mother_signal_r, mother_confidence_r)]
            infant_signals[part] = [(infant_signal_l, infant_confidence_l), (infant_signal_r, infant_confidence_r)]
        max_len = len(mother_signals["arm"][0][0])
        reses = analysis_sequence(mother_signals, infant_signals, fps = int(args.fps), min_seg_length=10)
        output_json_path = os.path.join(participant_output_path, file + "-segmented_analysis.json")
        output_img_path = os.path.join(participant_output_path, file + "-peak_curve.png")
        json.dump(reses, open(output_json_path, "w"))
        
        # plotting
        if not args.plot_off:
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
            metrics = [
                file,
                ppt,
                tp,
                tot_mother_init / tot_time,
                tot_infant_init / tot_time,
                tot_change / tot_time,
                np.var(np.array(all_peak_lags))
                ]
            
        except:
            print("Error with analysis parameters for this participant")
            metrics = [file, ppt, tp, np.nan, np.nan, np.nan, np.nan]

        with open(os.path.join(participant_output_path, file + "-metrics.csv"), 'w') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(metrics)

    return


if __name__ == '__main__':

    start = time.time()

    # Create the parser
    my_parser = argparse.ArgumentParser(description='Process some arguments')
    # Add the arguments
    my_parser.add_argument('fps',
                            type=str,
                            help='video fps, same for all videos')
    my_parser.add_argument('--plot_off',
                            help='create plots or not - otherwise just make segmented analysis json',
                            default=False,
                            action='store_true')

    # Execute the parse_args() method
    args = my_parser.parse_args()
    
    # set analysis folder with fps shown
    if not os.path.isdir(str(settings.MODEL1_FOLDER) + "_" + args.fps + "fps"): # make cross_corr output folder if does not exist
        os.mkdir(str(settings.MODEL1_FOLDER) + "_" + args.fps + "fps")

    # choose the camera with the most good frames for each participant and timepoint
    # implemented with multiprocess
    print('Data quality check:')
    checks = [
        ['LShoulder', 'LElbow'],
        ['RShoulder', 'RElbow'],
        ['Nose', 'LEar'],
        ['Nose', 'REar']
        ]
    camera_scores, best_cams = get_best_cams(checks)
    camera_scores.to_csv(settings.ANALYSIS_FOLDER / "camera_scores_model1.csv")
    best_cams.to_csv(str(settings.BEST_CAMERAS) + "_model1.csv")
    best_cam_list = list(zip(best_cams.Filename, best_cams.ppt, best_cams.tp))

    # run multicore
    print('Model 1 analysis:')
    pool = mp.Pool(mp.cpu_count())
    params = [(file, ppt, tp, args) for file, ppt, tp in best_cam_list]
    results = list(pool.starmap(model1, tqdm(params, total=len(params))))

    # output headers
    res_rows = [
        [
        "video",
        "participant",
        "timepoint",
        "mother_initiated_interactions",
        "infant_initiated_interactions",
        "change_of_leaders",
        "synchrony_stability"
        ]
        ]

    # read results from saved csvs
    for result in glob(str(settings.MODEL1_FOLDER) + "_" + args.fps + "fps" + "\\*\\*-metrics.csv"):
        with open(result, 'r', newline='') as csvfile:
            csv_reader = csv.reader(csvfile)
            res_rows.append(list(csv_reader)[0])

    # save as combined file
    with open(os.path.join(str(settings.MODEL1_FOLDER) + "_" + args.fps + "fps", "combined_results.csv"), "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerows(res_rows)

    end = time.time()
    print('Runtime: {}sec'.format(round(end - start)))

