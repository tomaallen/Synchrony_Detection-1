# %%
import os
from pathlib import Path
import numpy as np
from typing import Union
import argparse

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from data_quality_check import get_best_cams
from json2csv import read_json
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import settings

from multiSyncPy.synchrony_metrics import recurrence_matrix, rqa_metrics
from scipy.signal import savgol_filter



# %%
class Dyad: # pose entropy analysis object

	''' class for calculating pose synchrony metrics.
	Attributes: pos, conf, filtered
	'''

	def __init__(self, filepath: str, use_points: Union[np.array, list]):
		# import data from filepath		
		_adult, _infant = read_json(filepath)
		_adult = _adult.reshape(-1, 25, 3)
		_infant = _infant.reshape(-1, 25, 3)
		self.pos = np.stack((_adult[:,use_points,:2], _infant[:,use_points,:2])) \
					.transpose((3, 0, 1, 2)) # daydic positions: xy, person, timepoint, keypoint
		self.conf = np.stack((_adult[:,use_points,2], _infant[:,use_points,2])) # dyadic confidence

	def conf_filter(self, conf_level = 0.3):
		_mask = np.empty(self.pos.shape)
		_mask[:] = np.nan
		self.filtered = np.where(self.conf > conf_level, self.pos, _mask)
		# return self.pos

	def savgol(self, fps):
		self.smoothed = savgol_filter(self.filtered, window_length=fps*3, polyorder=2)
		# return self.smoothed

	def euclidean(self):
		_pos = self.smoothed.transpose((0, 2, 1, 3)) # xy, timepoint, person, keypoint
		_delta = _pos[:, 1:, :, :] - _pos[:, :-1, :, :]
		_delta = _delta[::2] # select every other velocity as implemented in original MATLAB script
		self.smoothed_speed = np.sqrt((_delta**2).sum(axis=0)) # timepoint, person, keypoint
		# return self.smoothed_speed

	def mdrqa_quality_check(self, fps, conf_level=0.3):
		 # TODO: create a filter for the needs of MRQA
		 # discard second if all nans?
		 # data is usable if there are x seconds in a row?
		
		return


# %%
if __name__ == '__main__':

	# Create the parser
	my_parser = argparse.ArgumentParser(description='Process some arguments')
	# Add the arguments
	my_parser.add_argument('fps',
							type=str,
							help='video fps, same for all videos')
	my_parser.add_argument('--radius',
							default=1,
							type=int,
							help='radius of MdRQA analysis')
	my_parser.add_argument('--normalise',
							action='store_false',
							help='normalise???') # TODO: check this is euc

	# Execute the parse_args() method
	args = my_parser.parse_args()

	# set analysis folder with fps shown
	settings.MODEL2_FOLDER = Path(str(settings.MODEL2_FOLDER) + "_" + args.fps + "fps")
	if not os.path.isdir(settings.MODEL2_FOLDER): # make MdRQA output folder if does not exist
		os.mkdir(settings.MODEL2_FOLDER)

    # choose the camera with the most good frames for each participant and timepoint
    # implemented with multiprocess
	print('Data quality check:')
	checks = [['Nose', 'Neck']]
	camera_scores, best_cams = get_best_cams(checks)
	camera_scores.to_csv(settings.ANALYSIS_FOLDER / "camera_scores_model1.csv")
	best_cams.to_csv(str(settings.BEST_CAMERAS) + "_model1.csv")
	best_cam_list = list(zip(best_cams.Filename, best_cams.ppt, best_cams.tp))

	# output headers
	use_points = [0, 1]  # use Nose and Neck only for analysis
	analysis_params = {
		'radius': 1,
		'normalise': True, # TODO: check this is euc
		'embedding_dimension': None,
		'embedding_delay': None
		} # TODO: make analysis params args
	res_rows = [["video", "position_recurrence", "velocity_recurrence"]] # TODO: add usability percentage?

	# TODO: add tqdm from code at bottom
	for file, ppt, tp in best_cam_list:
		print(file)
		
		json_path = settings.POSE_FOLDER / file / "json_files" / str(file + "-PD-combined_output.json")

		# prepare data from json file
		dyad = Dyad(json_path, use_points)
		dyad.conf_filter()
		dyad.savgol(args.fps)
		dyad.euclidean()

		# TODO: dyad.mdrqa_quality_check

		# TODO: take every other velocity and use smoothed values

		# TODO: optimise delay and embedding

		# reshape data to (n signals, timepoints)
		pos_input = dyad.smoothed.transpose((0, 1, 3, 2)).reshape(4*len(use_points), -1)
		vel_input = dyad.smoothed_speed.transpose((1, 2, 0)).reshape(2*len(use_points), -1)

		# run mdrqa with multiSyncPy
		pos_recur = recurrence_matrix(pos_input, **analysis_params)
		pos_rqa = rqa_metrics(pos_recur)

		vel_recur = recurrence_matrix(vel_input, **analysis_params)
		vel_rqa = rqa_metrics(vel_recur)

		# add results to output file
		res_rows.append([file, pos_rqa[0], vel_rqa[0]])


	# save results
	pd.DataFrame(res_rows).to_csv(settings.MODEL2_FOLDER / "mdrqa_results.csv")


# %%
# from run_model1.py - for multicore running

    # # run multicore
    # print('Model 1 analysis:')
    # pool = mp.Pool(mp.cpu_count())
    # params = [(file, ppt, tp, args) for file, ppt, tp in best_cam_list]
    # results = list(pool.starmap(model1, tqdm(params, total=len(params))))

    # # output headers
    # res_rows = [
    #     [
    #     "video",
    #     "participant",
    #     "timepoint",
    #     "mother_initiated_interactions",
    #     "infant_initiated_interactions",
    #     "change_of_leaders",
    #     "synchrony_stability"
    #     ]
    #     ]

    # # read results from saved csvs
    # for result in glob(str(settings.MODEL1_FOLDER) + "_" + args.fps + "fps" + "\\*\\*-metrics.csv"):
    #     with open(result, 'r', newline='') as csvfile:
    #         csv_reader = csv.reader(csvfile)
    #         res_rows.append(list(csv_reader)[0])

    # # save as combined file
    # with open(os.path.join(str(settings.MODEL1_FOLDER) + "_" + args.fps + "fps", "combined_results.csv"), "w", newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerows(res_rows)

    # end = time.time()
    # print('Runtime: {}sec'.format(round(end - start)))