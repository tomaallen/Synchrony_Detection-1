import csv
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# repository folder added to path in run_model1.py using:
# from pathlib import Path
# import sys
# sys.path.append(str(Path(os.getcwd()).parent))
from json2csv import read_json

keypoint_name_to_index = {"nose": 0, "lear": 17, "rear": 18, "lelbow": 6, "lshoulder": 5, "relbow": 3, "rshoulder": 2}

def interp_keypoint(data, keypoint_name, min_confidence = 0.6, max_interp_length = 20, min_avail_rate = 0.7):
	prev_avail_index = -1
	st_index = keypoint_name_to_index[keypoint_name] * 3
	interp_indices, avail_x, avail_y = [], [], []
	is_confident = np.ones(len(data))
	for i in range(len(data)):
		if data[i, st_index + 2] < min_confidence:
			interp_indices.append(i)
		else:
			avail_x.append(i)
			avail_y.append(data[i, st_index : st_index + 2])
			if i - prev_avail_index - 1 > max_interp_length:
				is_confident[prev_avail_index + 1 : i] = 0
			prev_avail_index = i
	if len(data) - prev_avail_index - 1 > max_interp_length:
		is_confident[prev_avail_index + 1 : ] = 0
	"""
	if len(avail_x) < min_avail_rate * len(data):
		return np.zeros((len(data), 2)), np.zeros(len(data))
	"""
	if len(avail_x) < 2:
		return np.zeros((len(data), 2)), np.zeros(len(data))
	fun = interp1d(avail_x, np.array(avail_y), axis = 0, fill_value = "extrapolate", assume_sorted = True)
	res_data = data[ : , st_index : st_index + 2].copy()
	interp_values = fun(interp_indices)
	res_data[interp_indices] = interp_values
	return res_data, is_confident

def get_head_angle(data, min_confidence = 0.6, max_interp_length = 20):
	if type(data) == str:
		data = read_csv(data)
	nose_points, nose_confidence = interp_keypoint(data, "nose", min_confidence, max_interp_length)
	lear_points, lear_confidence = interp_keypoint(data, "lear", min_confidence, max_interp_length)
	rear_points, rear_confidence = interp_keypoint(data, "rear", min_confidence, max_interp_length)
	lear_vector, rear_vector = lear_points - nose_points, rear_points - nose_points
	langle, rangle = np.arctan2(lear_vector[ : , 0], lear_vector[ : , 1]), np.arctan2(rear_vector[ : , 0], rear_vector[ : , 1])
	return langle, (nose_confidence + lear_confidence) / 2, rangle, (nose_confidence + rear_confidence) / 2

def get_arm_angle(data, min_confidence = 0.6, max_interp_length = 20):
	if type(data) == str:
		data = read_csv(data)
	lshoulder_points, lshoulder_confidence = interp_keypoint(data, "lshoulder", min_confidence, max_interp_length)
	rshoulder_points, rshoulder_confidence = interp_keypoint(data, "rshoulder", min_confidence, max_interp_length)
	lelbow_points, lelbow_confidence = interp_keypoint(data, "lelbow", min_confidence, max_interp_length)
	relbow_points, relbow_confidence = interp_keypoint(data, "relbow", min_confidence, max_interp_length)
	larm_vector, rarm_vector = lelbow_points - lshoulder_points, relbow_points - rshoulder_points
	langle, rangle = np.arctan2(larm_vector[ : , 0], larm_vector[ : , 1]), np.arctan2(rarm_vector[ : , 0], rarm_vector[ : , 1])
	return langle, (lshoulder_confidence + lelbow_confidence) / 2, rangle, (rshoulder_confidence + relbow_confidence) / 2

def extract_confident_segments(confidence, confidence_threshold = 0.8, min_seg_length = 500):
    start_conf, end_conf = -2, -2
    res_segments = []
    for i in range(len(confidence)):
        if confidence[i] > confidence_threshold:
            if end_conf != i - 1:
                if end_conf - start_conf + 1 >= min_seg_length:
                    res_segments.append((start_conf, end_conf + 1))
                start_conf = i
            end_conf = i
    if end_conf - start_conf + 1 >= min_seg_length:
        res_segments.append((start_conf, end_conf + 1))
    return res_segments

def read_csv(path):
	data = []
	with open(path, newline = "") as f:
		reader = csv.reader(f)
		for i, row in enumerate(reader):
			if i > 0:
				data.append([float(x) for x in row])
	return np.array(data)

if __name__ == '__main__':
	#data = read_csv("02_S001_02_op_mom.csv")
	data, data2 = read_json("./Openpose outputs/Singapore Data/LP004_PCI/json_files/LP004_PCI-PD-combined_output.json")
	langle, lconfidence, rangle, rconfidence = get_head_angle(data, min_confidence = 0.1)
	plt.plot(langle, label = "Left ear")
	plt.plot(rangle, label = "Right ear")
	plt.plot(lconfidence, label = "L confidence")
	plt.plot(rconfidence, label = "R confidence")
	plt.title("Head position curve")
	plt.legend()
	plt.show()
	print(lconfidence.shape, rconfidence.shape)
	print(extract_confident_segments(lconfidence))
	print(extract_confident_segments(rconfidence))
	langle, lconfidence, rangle, rconfidence = get_arm_angle(data, min_confidence = 0.1)
	plt.plot(langle, label = "Left arm")
	plt.plot(rangle, label = "Right arm")
	plt.plot(lconfidence, label = "L confidence")
	plt.plot(rconfidence, label = "R confidence")
	plt.title("Arm position curve")
	plt.legend()
	plt.show()
	print(extract_confident_segments(lconfidence))
	print(extract_confident_segments(rconfidence))
