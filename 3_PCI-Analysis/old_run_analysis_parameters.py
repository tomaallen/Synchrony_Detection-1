import numpy as np
import os
import csv
import json
from model1.corr_functions import *

if __name__ == '__main__':

	## Computing Synchrony parameters	

	res_rows = [["site", "participant", "mother_initiated_interactions", "infant_initiated_interactions", "change_of_leaders", "intensity(variance)"]]

	for site_name, fps in [("Singapore", 25.0), ("Brazil", 29.97), ("Bangladesh", 29.97)]:
		site_path = os.path.join("./Openpose outputs", f"{site_name} Data")
		for participant_name in os.listdir(site_path):
			json_path = os.path.join(os.path.join(site_path, participant_name), "json_files")
			participant_output_path = os.path.join(site_path.replace("Openpose", "Analysis"), participant_name)
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
					res_rows.append([site_name, comb_json_path.replace("-combined_output.json", ""), tot_mother_init / tot_time, tot_infant_init / tot_time, tot_change / tot_time, np.var(np.array(all_peak_lags))])

	with open("normalized_results.csv", "w", newline = "") as f:
		writer = csv.writer(f)
		writer.writerows(res_rows)
					
	exit()

	