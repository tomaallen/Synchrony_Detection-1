import numpy as np
import os
import csv
import json
from model1.corr_functions import *
import argparse

if __name__ == '__main__':
    # Create the parser
    my_parser = argparse.ArgumentParser(description='Process some arguments')

    # Add the arguments
    my_parser.add_argument('--PCI_dir',
							type=str,
							help='output folder for PCI combined analysis',
							default=r'')
    my_parser.add_argument('--reach_dir',
							type=str,
							help='video specific folder of pose detection outputs',
							default=r'')

	# Execute the parse_args() method
    args = my_parser.parse_args()
    res_rows = [["participant", "mother_initiated_interactions", "infant_initiated_interactions", "change_of_leaders", "intensity(variance)"]]
        
    # create relevant paths from video name
    vid = os.path.basename(args.reach_dir)
    json_path = os.path.join(args.reach_dir, "json_files")
    participant_output_path = os.path.join(args.PCI_dir, vid)
    os.makedirs(participant_output_path, exist_ok = True)
    
    # flag that run plots needs to run first if PCI directory does not exist
    if not os.path.exists(os.path.join(participant_output_path, vid + "-segmented_analysis.json")):
        raise Exception('Need to run run_analysis_plots.py first')
    
    # run analysis parameters
    for comb_json_path in os.listdir(json_path):
        if comb_json_path.endswith("combined_output.json"):
            output_json_path = os.path.join(participant_output_path, vid + "-segmented_analysis.json")
            reses = json.load(open(output_json_path))
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
                res_rows.append([comb_json_path.replace("-PD-combined_output.json", ""), tot_mother_init / tot_time, tot_infant_init / tot_time, tot_change / tot_time, np.var(np.array(all_peak_lags))])
            except:
                print("Error with analysis parameters for this participant")
                res_rows.append([comb_json_path.replace("-combined_output.json", ""), "zero error", "zero error", "zero error", "zero error"])

    with open(os.path.join(participant_output_path, vid + "-normalized_results.csv"), "w", newline = "") as f:
        writer = csv.writer(f)
        writer.writerows(res_rows)
    
    exit()
