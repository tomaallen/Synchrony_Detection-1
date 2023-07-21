import numpy as np
import json
import jsonlines

keypoints_dict = {0: "Nose",
                  1: "Neck",
                  2: "RShoulder",
                  3: "RElbow",
                  4: "RWrist",
                  5: "LShoulder",
                  6: "LElbow",
                  7: "LWrist",
                  8: "MidHip",
                  9: "RHip",
                  10: "RKnee",
                  11: "RAnkle",
                  12: "LHip",
                  13: "LKnee",
                  14: "LAnkle",
                  15: "REye",
                  16: "LEye",
                  17: "REar",
                  18: "LEar",
                  19: "LBigToe",
                  20: "LSmallToe",
                  21: "LHeel",
                  22: "RBigToe",
                  24: "RHeel"}

frames = {}

for no_frames in range(2002):
    print(no_frames)
    no_persons = 0
    data = np.genfromtxt("save_csv\point" + str(no_frames) + ".csv", delimiter=",")
    # print(data)
    # input()
    if data.ndim <= 1:
        data_2 = np.expand_dims(data, axis=0)
        no_persons = 1
    else:
        data_2 = data
        no_persons = np.shape(data)[0]
    # print(no_persons)

    data_id = np.genfromtxt("save_csv\_track" + str(no_frames) + ".csv", delimiter=",")
    # print(data_2)
    # print(data_id)
    
    

    # dict_person = dict(enumerate(data_2.tolist(), 1))
    dict_person = {}
    try:
        for _id, _data in zip(data_id,data_2.tolist()):
            dict_person[int(_id)] = _data
    except:
        dict_person[int(data_id)] = data_2.tolist()[0]
    # print(dict_person)
    # print(dict_person_)

    # input()

    new_dict_persons = {}
    for k, v in dict_person.items():
        keypoins_list = []
        for i in range(0, len(v), 3):
            keypoins_list.append(v[i:i+3])
        keypoins_dict_ = dict(enumerate(keypoins_list))
        keypoins_dict_final = dict((value, keypoins_dict_[key]) for (key, value) in keypoints_dict.items())
        # print(keypoins_dict_final)
        new_dict_persons[k] = keypoins_dict_final

    subdict_of_frame = {"Count": no_persons, "Data": new_dict_persons}    
    frames[no_frames+1] = subdict_of_frame
    # print(frames)
    # input()
    # print(len(frames))

with open('Camcorder 2 Demo_output.jsonl', 'w') as fp:
    json.dump(frames, fp)
    # fp.write('\n')

# with jsonlines.open('Camcorder 1 Demo_output.jsonl', 'w') as writer:
#     writer.write_all(frames)



