# %%
import os
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from data_quality_check import get_best_cams
from json2csv import read_json
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import settings

import matlab.engine


# %%
if __name__ == '__main__':

	# Create the parser
	my_parser = argparse.ArgumentParser(description='Process some arguments')
	# Add the arguments
	my_parser.add_argument('fps',
							type=str,
							help='video fps, same for all videos')

	# Execute the parse_args() method
	args = my_parser.parse_args()

# class arguments():
# 	def __init__(self):
# 		self.fps='30'

# args = arguments()

	# set analysis folder with fps shown
	if not os.path.isdir(str(settings.MODEL2_FOLDER) + "_" + args.fps + "fps"): # make MdRQA output folder if does not exist
		os.mkdir(str(settings.MODEL2_FOLDER) + "_" + args.fps + "fps")

	# choose the camera with the most good frames for each participant and timepoint
	# implemented with multiprocess
	# print('Data quality check:')
	# checks = [['Nose', 'Neck']]
	# camera_scores, best_cams = get_best_cams(checks)
	# camera_scores.to_csv(settings.ANALYSIS_FOLDER / "camera_scores_model2.csv")
	# best_cams.to_csv(str(settings.BEST_CAMERAS) + "_model2.csv")
	best_cams = pd.read_csv(str(settings.BEST_CAMERAS) + "_model2.csv", index_col=0)
	best_cam_list = list(zip(best_cams.Filename, best_cams.ppt, best_cams.tp))

	# make matlab inputs directory
	os.makedirs(str(settings.MODEL2_FOLDER) + "_" + args.fps + "fps" + "\\matlab_inputs", exist_ok=True)

	# output headers
	use_points = [0, 1]  # use Nose and Neck only for analysis
	input_headers = [
		"X_Neck_b",
		"Y_Neck_b",
		"X_Nose_b",
		"Y_Nose_b",
		"X_Neck_m",
		"Y_Neck_m",
		"X_Nose_m",
		"Y_Nose_m"
	]

	for file, ppt, tp in best_cam_list:
		if not os.path.isfile(str(settings.MODEL2_FOLDER) + "_" + args.fps + "fps" + f"\\matlab_inputs\\t{tp}\\" + str(file + ".csv")):
			# read combined json data file
			print(file)
			json_path = settings.FOLDER / "json_files" / str(file + ".json") # editted for json only
			adult, infant = read_json(json_path)

			# read frame check file - conf threshold set in combined_analysis.py
			frame_check = pd.read_csv(settings.FRAME_CHECKS / (file + ".csv"), index_col=0)

			# set bad frames to nan using frame_check
			# bad frames are frames with no dyad present, or confidence too low for any key-points being used
			adult[(frame_check.DyadPresent == True) & (frame_check.Nose == True) & (frame_check.Neck == True)] = np.nan
			infant[(frame_check.DyadPresent == True) & (frame_check.Nose == True) & (frame_check.Neck == True)] = np.nan

			# want columns 0,1,3,4 = Nose_x, Nose_y, Neck_x, Neck_y
			matlab_input = pd.DataFrame(
				np.concatenate((adult[:,[0,1,3,4]], infant[:,[0,1,3,4]]), axis=1),
				columns=input_headers
			).replace(
				np.nan,
				'nan'
			)

			# save MATLAB inputs into a single folder
			matlab_input.to_csv(
				str(settings.MODEL2_FOLDER) + "_" + args.fps + "fps" + f"\\matlab_inputs\\t{tp}\\" + str(file + ".csv"),
				index=False
			)

	# run mdrqa using matlab engine
	for tp in ['3m', '6m', '12m']:
		eng = matlab.engine.start_matlab()
		eng.cd(os.getcwd())
		_ = eng.run_mdrqa(
			str(settings.MODEL2_FOLDER) + "_" + args.fps + "fps" + f"\\matlab_inputs\\t{tp}\\", # input folder
			str(settings.MODEL2_FOLDER) + "_" + args.fps + "fps" + f"\\mdrqa_results_t{tp}.csv", # output folder
			int(args.fps), # fps
			35, # usability threshold,
			0.3, # radius threshold
			100, # max lag
			10 # max embedding dimensions
		)

	# results = pd.read_csv(str(settings.MODEL2_FOLDER) + "_" + args.fps + "fps\\mdrqa_results.csv")
	# results['Filename'] = results.ID.apply(lambda x : os.path.splitext(x)[0])
	# results = results.merge(best_cams, on='Filename').drop(columns=['ID', 'QualityScore'])
	# results = results.loc[:,list(results.columns[3:]) + list(results.columns[:3])]
	# results.to_csv(str(settings.MODEL2_FOLDER) + "_" + args.fps + "fps\\mdrqa_results.csv", index=False)

