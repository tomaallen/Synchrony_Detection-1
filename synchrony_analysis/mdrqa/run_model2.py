# %%
import os
import numpy as np
from typing import Union
import argparse

from pathlib import Path
import sys
sys.path.append(str(Path(os.getcwd()).parent))
import settings
import data_quality_check as dqc
from json2csv import read_json

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

	def euclidean(self):
		_pos = self.pos.transpose((0, 2, 1, 3)) # xy, timepoint, person, keypoint
		_delta = _pos[:, 1:, :, :] - _pos[:, :-1, :, :]
		self.speed = np.sqrt((_delta**2).sum(axis=0)) # timepoint, person, keypoint
		# return self.speed

	def sav_gol(self):
		savgol_filter(self.pos, window_length=settings.FPS * 3, )

	
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

	# Execute the parse_args() method
	args = my_parser.parse_args()

	# set analysis folder with fps shown
	settings.MODEL2_FOLDER = Path(str(settings.MODEL2_FOLDER) + "_" + args.fps + "fps")
	if not os.path.isdir(settings.MODEL2_FOLDER): # make MdRQA output folder if does not exist
		os.mkdir(settings.MODEL2_FOLDER)

	# choose the camera with the most good frames for each participant and timepoint
	best_cam_list = dqc.get_best_cams([['Nose', 'Neck']])
	best_cam_list.to_csv(str(settings.BEST_CAMERAS) + "_model2.csv")

	# output headers
	use_points = [0, 1]  # FIXME: use Nose and Neck only for analysis - make it an arg?
	analysis_params = {'radius': 1,
						'normalise': True, # TODO: check this is euc
						'embedding_dimension': None,
						'embedding_delay': None} # TODO: make analysis params args
	res_rows = [["video",
				"position_recurrence",
				"velocity_recurrence"]] # TODO: add usability percentage?

	for file in best_cam_list:
		
		json_path = os.path.join(settings.POSE_FOLDER, file, "json_files", file + "-PD-combined_output.json")

		# prepare data from json file
		dyad = Dyad(json_path, use_points)
		dyad.conf_filter()
		dyad.euclidean()
		dyad.savgol()
		# dyad.filtered

		# TODO: dyad.mdrqa_quality_check

		# reshape data to (n signals, timepoints)
		pos_input = dyad.pos.transpose((0, 1, 3, 2)).reshape(4*len(use_points), -1)
		vel_input = dyad.speed.transpose((1, 2, 0)).reshape(2*len(use_points), -1)

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
