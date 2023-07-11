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
							help='the file(s) or folder of input images and/or videos',
							default=r'C:\Users\labadmin\Desktop\SynchronyAnalysis-Upload\3_PCI-Analysis')
	my_parser.add_argument('--reach_dir',
							type=str,
							help='the file(s) or folder of output images and/or videos',
							default=r'Y:\Synchronised_trimmed_videos_STT\Pose_detect_output\Mom_Cam')

	# Execute the parse_args() method
	args = my_parser.parse_args()

	res_rows = [["site", "participant", "mother_initiated_interactions", "infant_initiated_interactions", "change_of_leaders", "intensity(variance)"]]

	for site_name, fps in [("Brazil", 25.0)]: #, ("Brazil", 29.97), ("Bangladesh", 29.97)]:
		for participant_file in os.listdir(args.reach_dir):
			# adapt it to the name of the files
			participant_name = str.split(participant_file,'_')[0] # Tom edit
			print(participant_name)
			json_path = os.path.join(os.path.join(args.reach_dir, participant_file), "json_files")
			participant_output_path = os.path.join(os.path.join(args.PCI_dir, "Full_Analysis_Outputs"), participant_name)
			os.makedirs(participant_output_path, exist_ok = True)
			for comb_json_path in os.listdir(json_path):
				if comb_json_path.endswith("combined_output.json"):
					output_json_path = os.path.join(participant_output_path, "segmented_analysis.json")
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
						res_rows.append([site_name, comb_json_path.replace("-combined_output.json", ""), tot_mother_init / tot_time, tot_infant_init / tot_time, tot_change / tot_time, np.var(np.array(all_peak_lags))])
					except:
						print("Error with this participant")
						res_rows.append([site_name, comb_json_path.replace("-combined_output.json", ""), "zero error", "zero error", "zero error", "zero error"])

	with open(os.path.join(args.PCI_dir, "normalized_results.csv"), "w", newline = "") as f:
		writer = csv.writer(f)
		writer.writerows(res_rows)
					
	exit()

	