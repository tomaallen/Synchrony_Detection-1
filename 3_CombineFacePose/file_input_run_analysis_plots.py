import numpy as np
import matplotlib.pyplot as plt
from model1.preprocess import get_head_angle, get_arm_angle
import os
import json
from model1.json2csv import read_json
from model1.corr_functions import *
import argparse

if __name__ == '__main__':
    
    # Create the parser
    my_parser = argparse.ArgumentParser(description='Process some arguments')

    # TODO: add args here for gridsearch
    # Add the arguments
    my_parser.add_argument('--PCI_dir',
							type=str,
							help='folder for combined analysis output',
							default=r'D:\\BRAINRISE_PCI\\Downloaded\\cross_corr_output')
    my_parser.add_argument('--reach_dir',
							type=str,
							help='reaching detection output folder for the given video',
							default=r'D:\\BRAINRISE_PCI\\Downloaded\\pose_detect_output\\006_03-06-22_2a_av__c4')
    my_parser.add_argument('--fps',
							type=int,
							help='video fps',
							default=25.00)
    my_parser.add_argument('--plot_on',
							type=bool,
							help='create plots or not - otherwise just make segmented analysis json',
							default=True)
    # add arg for analysis parameters

	# Execute the parse_args() method
    args = my_parser.parse_args()
    
    # make PCI directory if it does not exist - already implemented in combined_analysis.py
    if not os.path.exists(args.PCI_dir):
        os.mkdir(args.PCI_dir)

	## Correlation analysis: Saving plots
    vid = os.path.basename(args.reach_dir)
    #print(vid)
    json_path = os.path.join(args.reach_dir, "json_files")
    participant_output_path = os.path.join(args.PCI_dir, vid)
    os.makedirs(participant_output_path, exist_ok = True) # make participant folder in PCI dir if does not exist
    
    # if not os.path.exists(os.path.join(participant_output_path, vid + "-segmented_analysis.json")): # run only if not run before
    for comb_json_path in os.listdir(json_path):
        if comb_json_path.endswith("combined_output.json"):
            
            # json segmented analysis
            # TODO: separate this from plot so parameters can be run without plots
            comb_json_path = os.path.join(json_path, comb_json_path)
            mother_tot, infant_tot = read_json(comb_json_path)
            mother_signals, infant_signals = {}, {}
            for part, func in [("arm", get_arm_angle), ("head", get_head_angle)]:
                mother_signal_l, mother_confidence_l, mother_signal_r, mother_confidence_r = func(mother_tot, 0.1, int(round(3 * args.fps)))
                infant_signal_l, infant_confidence_l, infant_signal_r, infant_confidence_r = func(infant_tot, 0.1, int(round(3 * args.fps)))
                mother_signals[part] = [(mother_signal_l, mother_confidence_l), (mother_signal_r, mother_confidence_r)]
                infant_signals[part] = [(infant_signal_l, infant_confidence_l), (infant_signal_r, infant_confidence_r)]
            max_len = len(mother_signals["arm"][0][0])
            reses = analysis_sequence(mother_signals, infant_signals, fps = args.fps, min_seg_length=10) # TODO: add args here for gridsearch
            output_json_path = os.path.join(participant_output_path, vid + "-segmented_analysis.json")
            output_img_path = os.path.join(participant_output_path, vid + "-peak_curve.png")
            json.dump(reses, open(output_json_path, "w"))
            
            # plotting
            if args.plot_on:
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
                #plt.show()
                plt.savefig(output_img_path)
                plt.close(fig)

    exit()

	