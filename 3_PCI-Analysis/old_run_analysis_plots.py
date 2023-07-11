import numpy as np
import matplotlib.pyplot as plt
from model1.preprocess import get_head_angle, get_arm_angle
import os
import json
from model1.json2csv import read_json
from model1.corr_functions import *

if __name__ == '__main__':

	## Correlation analysis: Saving plots

	for site_name, fps in [("Singapore", 25.0), ("Brazil", 29.97), ("Bangladesh", 29.97)]:
		site_path = os.path.join("./Openpose outputs", f"{site_name} Data")
		for participant_name in os.listdir(site_path):
			json_path = os.path.join(os.path.join(site_path, participant_name), "json_files")
			participant_output_path = os.path.join(site_path.replace("Openpose", "Analysis"), participant_name)
			os.makedirs(participant_output_path, exist_ok = True)
			for comb_json_path in os.listdir(json_path):
				if comb_json_path.endswith("combined_output.json"):
					comb_json_path = os.path.join(json_path, comb_json_path)
					mother_tot, infant_tot = read_json(comb_json_path)
					mother_signals, infant_signals = {}, {}
					for part, func in [("arm", get_arm_angle), ("head", get_head_angle)]:
						mother_signal_l, mother_confidence_l, mother_signal_r, mother_confidence_r = func(mother_tot, 0.1, int(round(3 * fps)))
						infant_signal_l, infant_confidence_l, infant_signal_r, infant_confidence_r = func(infant_tot, 0.1, int(round(3 * fps)))
						mother_signals[part] = [(mother_signal_l, mother_confidence_l), (mother_signal_r, mother_confidence_r)]
						infant_signals[part] = [(infant_signal_l, infant_confidence_l), (infant_signal_r, infant_confidence_r)]
					max_len = len(mother_signals["arm"][0][0])
					reses = analysis_sequence(mother_signals, infant_signals, fps = fps)
					output_json_path = os.path.join(participant_output_path, "segmented_analysis.json")
					output_img_path = os.path.join(participant_output_path, "peak_curve.png")
					json.dump(reses, open(output_json_path, "w"))
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
					plt.show()

	
	exit()

	