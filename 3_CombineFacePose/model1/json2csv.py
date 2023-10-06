import json
import numpy as np
#from visualization import show_sequence

id_to_name = {0: "Nose", 1: "Neck", 2: "RShoulder", 3: "RElbow", 4: "RWrist", 5: "LShoulder", 6: "LElbow", 7: "LWrist", 8: "MidHip", 9: "RHip", 10: "RKnee", 11: "RAnkle", 12: "LHip", 13: "LKnee", 14: "LAnkle", 15: "REye", 16: "LEye", 17: "REar", 18: "LEar", 19: "LBigToe", 20: "LSmallToe", 21: "LHeel", 22: "RBigToe", 23: "RSmallToe", 24: "RHeel"}
def read_json(path):
	with open(path) as f:
		data = json.load(f)
	mother_data, infant_data = [], []
	i = 1
	cts = {0: 0, 1: 0, -1: 0}
	unknown_skeletons, failed_infant, failed_mother = 0, 0, 0
	while str(i) in data:
		t_f = data[str(i)]
		got_mother, got_infant = False, False
		for t_s in t_f["Data"]:
			t_kps = []
			for k, kname in id_to_name.items():
				t_kps.append(t_s["Body"][kname])
			t_kps = np.array(t_kps)
			cts[t_s["baby_ma_id"]] += 1
			if t_s["baby_ma_id"] == 0:
				mother_data.append(t_kps)
				got_mother = True
			elif t_s["baby_ma_id"] == 1:
				infant_data.append(t_kps)
				got_infant = True
			else:
				unknown_skeletons += 1
		if not got_mother:
			mother_data.append(np.zeros((25, 3)))
			failed_mother += 1
		if not got_infant:
			infant_data.append(np.zeros((25, 3)))
			failed_infant += 1
		i += 1
	return np.stack(mother_data).reshape(-1, 25 * 3), np.stack(infant_data).reshape(-1, 25 * 3)

if __name__ == '__main__':
	x1, x2 = read_json("./Openpose outputs/Singapore Data/LP004_PCI/json_files/LP004_PCI-PD-combined_output.json")
	x1 = x1.reshape(len(x1), 25, 3)
	x2 = x2.reshape(len(x2), 25, 3)
	#print(x1[4000 : , 0, 2], x1[4000 : , 17, 2], x1[4000 : , 18, 2])
	print(np.mean(x2[4000 : , 0, 2]), np.mean(x2[4000 : , 17, 2]), np.mean(x2[4000 : , 18, 2]))