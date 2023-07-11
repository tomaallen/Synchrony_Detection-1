import copy
import json
import numpy as np
import os
import reaching_const
import utils
from utils import ComplexEncoder
from human_properties import Person, average_multi_person_ma, body_base_distance
import queue


def compare_head_diagonal_ratios(*ratios_lists):
    temp_ = ratios_lists[0]
    for i in range(1, len(ratios_lists)):
        temp_ = np.vstack((temp_, ratios_lists[i]))
    normalized_temp_ = utils.normalized(temp_, 0)
    _score = np.nansum(normalized_temp_, axis=1)
    return _score


def compute_total_score(frame, average_baby, average_mom, list_of_person, ratio_scores=None, list_of_ratio_confs=None):
    no_person = len(list_of_person)
    ratio_total_scores = [0.] * no_person
    total_scores = [0.] * no_person
    baby_dist_score = [0.] * no_person
    mom_dist_score = [0.] * no_person
    baby_ratio_score = [0.] * no_person
    mom_ratio_score = [0.] * no_person
    dist_to_baby = []
    dist_to_mom = []
    if average_baby is not None and average_mom is not None:
        for _person in list_of_person:
            dist_to_baby.append(body_base_distance(_person, average_baby))
            dist_to_mom.append(body_base_distance(_person, average_mom))

        if no_person == 1:
            if dist_to_baby[0] < dist_to_mom[0]:
                baby_dist_score[0] = 1.25
            elif dist_to_baby[0] >= dist_to_mom[0]:
                mom_dist_score[0] = 1.25
        elif no_person >= 2:
            baby_ = np.argmin(dist_to_baby)
            mom_ = np.argmin(dist_to_mom)
            baby_dist_score[baby_] = 1.25
            mom_dist_score[mom_] = 1.25
    # if baby_ == mom_:
    #     if dist_to_baby[baby_] < dist_to_mom[mom_]:
    #         baby_score[baby_] = 1
    #         mom_score[baby_] = 0
    # if 30 < int(frame) < 76:
    #     print('baby: dist to ave baby', dist_to_baby, baby_dist_score)
    #     print('mom: dist to ave mom', dist_to_mom, mom_dist_score)
    closeness_scores_ = [baby_dist_score, mom_dist_score]
    closeness_scores = zip(baby_dist_score, mom_dist_score)
    # print(closeness_scores_)
    closeness_scores = [_score[0] - _score[1] for _score in closeness_scores]

    if ratio_scores is None:
        total_scores = closeness_scores
        return closeness_scores_, closeness_scores, ratio_total_scores, total_scores
    else:

        _mom = np.argmax(ratio_scores)
        _baby = np.argmin(ratio_scores)
        if list_of_ratio_confs[_mom] >= list_of_ratio_confs[_baby]:
            baby_ratio_score[_baby] = 1
            mom_ratio_score[_mom] = 1
        else:
            baby_ratio_score[_baby] = 0.5
            mom_ratio_score[_mom] = 0.5
    ratio_total_scores = zip(baby_ratio_score, mom_ratio_score)
    ratio_total_scores = [_score[0] - _score[1] for _score in ratio_total_scores]
    # print('ratio', ratio_total_scores)

    total_scores = zip(closeness_scores, ratio_total_scores)
    total_scores = [_score[0] + _score[1] for _score in total_scores]
    # print('total', total_scores)
    # input()
    # if int(frame) == 96:
    #     print(ratio_scores, )

    return closeness_scores_, closeness_scores, ratio_total_scores, total_scores


def get_score_for_persons_ma(list_of_frames, json_data, json_score_only, window=10):
    several_frame_queue_baby = queue.Queue()
    several_frame_queue_mom = queue.Queue()
    scores_frames = {}
    the_queues = [several_frame_queue_baby, several_frame_queue_mom]
    for _frame, _data in json_data.items():
        # print('frame:', _frame)
        if int(_frame) in list_of_frames:
            # print('====================================================')
            print('frame number:', _frame)
            # if 3800 < int(_frame) < 3820:
            #     print('===========================================================================')
            #     print(_frame)
            list_of_person = []
            list_of_person_ma = []
            for _person_id, _pose in _data['Data'].items():
                person_temp = Person()
                person_temp.input_skeleton(_person_id, _pose)
                list_of_person.append(person_temp)

            ratio_scores = None
            list_of_ratio_confs = None
            if len(list_of_person) >= 2:
                list_of_ratios = []
                list_of_ratio_confs = []
                for _person in list_of_person:
                    # print('ratios for', _person.ID)
                    temp_, conf_ = _person.Body.head_diagonal_ratios()
                    # print(temp_)
                    list_of_ratios.append(temp_)
                    list_of_ratio_confs.append(conf_)

                ratio_scores = compare_head_diagonal_ratios(list_of_ratios)

            average_baby, len_baby = average_multi_person_ma(*list(the_queues[0].queue)), len(the_queues[0].queue)
            # average_baby.baby_MA_ID = 1
            average_mom, len_mom = average_multi_person_ma(*list(the_queues[1].queue)), len(the_queues[1].queue)
            # average_mom.baby_MA_ID = 0
            # if 3800 < int(_frame) < 3820:
            #     xlist = [json.dumps(_person, cls=ComplexEncoder) for _person in list(the_queues[1].queue)]
            #     for _l in xlist:
            #         print(_l)
            #     print('ave baby', json.dumps(average_baby, cls=ComplexEncoder))
            #     print('ave mom', json.dumps(average_mom, cls=ComplexEncoder))
            #     input()

            closeness_scores_, closeness_scores, ratio_total_scores, total_score \
                = compute_total_score(_frame, average_baby,
                                      average_mom,
                                      list_of_person,
                                      ratio_scores,
                                      list_of_ratio_confs)

            max_score = np.max(total_score)
            min_score = np.min(total_score)
            # print('max score is', max_score, 'min score is', min_score)
            for i, _person in enumerate(list_of_person):
                # print('ratios for', _person.ID)
                _person.total_score = total_score[i]
                _person.closeness_score = closeness_scores[i]
                _person.head_ratio_score = ratio_total_scores[i]
                if total_score[i] == max_score and max_score > 0:
                    _person.baby_MA_ID = 1
                    if len(several_frame_queue_baby.queue) < window:
                        several_frame_queue_baby.put(_person)
                        # print('baby: init put')
                    else:
                        several_frame_queue_baby.get()
                        several_frame_queue_baby.put(_person)
                        # print('baby: subsequent get and put')

                elif total_score[i] == min_score and min_score < 0:
                    _person.baby_MA_ID = 0
                    if len(several_frame_queue_mom.queue) < window:
                        several_frame_queue_mom.put(_person)
                        # print('mom: init put')
                    else:
                        several_frame_queue_mom.get()
                        several_frame_queue_mom.put(_person)
                        # print('mom: subsequent get and put')

            # if max_score <= 0:
            #     person_temp = Person()
            #     person_temp.Body.body_nan()
            #     if len(several_frame_queue_baby.queue) < window:
            #         several_frame_queue_baby.put(person_temp)
            #         # print('baby NaN: init put, ')
            #     else:
            #         several_frame_queue_baby.get()
            #         several_frame_queue_baby.put(person_temp)
            #         # print('baby NaN: subsequent get and put')
            # if min_score >= 0:
            #     person_temp = Person()
            #     person_temp.Body.body_nan()
            #     if len(several_frame_queue_mom.queue) < window:
            #         several_frame_queue_mom.put(person_temp)
            #         # print('mom NaN: init put')
            #     else:
            #         several_frame_queue_mom.get()
            #         several_frame_queue_mom.put(person_temp)
            #         # print('mom NaN: subsequent get and put')

                # print(scores[i])
            if json_score_only == 1:
                new_dict_persons = {}
                for k, _person in enumerate(list_of_person):
                    new_dict_persons[_person.ID] = [_person.closeness_score, _person.head_ratio_score,
                                                    _person.total_score, _person.baby_MA_ID]
                subdict_of_frame = {"Count": _data["Count"], "Data": new_dict_persons}
                scores_frames[_frame] = subdict_of_frame
            elif json_score_only == 2:
                new_dict_persons = {}
                for k, _person in enumerate(list_of_person):
                    new_dict_persons[_person.ID] = {"Neck": _person.Body.Neck.save_into_array().tolist(),
                                                    "close": _person.closeness_score,
                                                    "ratio": _person.head_ratio_score,
                                                    "total": _person.total_score,
                                                    "ma_id": _person.baby_MA_ID}

                if average_baby is not None:
                    new_dict_persons["-22"] = {"Neck": average_baby.Body.Neck.save_into_array().tolist(),
                                               "len": len_baby}
                else:
                    new_dict_persons["-22"] = {"Neck": None}
                if average_mom is not None:
                    new_dict_persons["-55"] = {"Neck": average_mom.Body.Neck.save_into_array().tolist(),
                                               "len": len_mom}
                else:
                    new_dict_persons["-55"] = {"Neck": None}
                subdict_of_frame = {"Count": _data["Count"], "Min score": min_score, "Max score": max_score,
                                    "Closeness": [list(closeness_scores_), closeness_scores],
                                    "Data": new_dict_persons}
                # if 30 < int(_frame) < 76:
                #     # print(_person.ID, new_dict_persons[_person.ID])
                #     print(subdict_of_frame)
                # elif int(_frame) > 76:
                #     input()
                # input()
                scores_frames[_frame] = subdict_of_frame
            else:
                new_dict_persons = []
                for k, _person in enumerate(list_of_person):
                    __person = copy.deepcopy(_person)
                    __person.Body.body_zero()
                    new_dict_persons.append(json.loads(json.dumps(__person, cls=ComplexEncoder)))
                    # if 30 < int(_frame) < 76:
                    #     # print('=======================================')
                    #     # print(_frame)
                    #     print(json.dumps(__person, cls=ComplexEncoder))
                    #     print("close", __person.closeness_score,
                    #           "ratio", __person.head_ratio_score,
                    #           "total", __person.total_score,
                    #           "ma_id", __person.baby_MA_ID)
                subdict_of_frame = {"Count": _data["Count"], "Data": new_dict_persons}
                scores_frames[_frame] = subdict_of_frame
            # input()
    return scores_frames


def poses_for_frames(list_of_frames=None, output_json_folder=None, json_score_only=1):
    # arguments:
    # list_of_frames: Eg. range(10, 20)
    # list all file paths in folder "json_files"

    json_file = utils.file_path_giving_folder("json_files")
    # load the json file into a dictionary
    for _file in json_file:
        if reaching_const.JSON_FILE in _file:
            print(_file)
            with open(_file, 'r') as f:
                json_data = json.load(f)

    no_csv_files = len(json_data)
    # print(no_csv_files)
    if list_of_frames is None:
        list_of_frames = range(0 + 1, no_csv_files + 1)

    print(list_of_frames)
    # input()
    window = 10

    scores_frames = get_score_for_persons_ma(list_of_frames, json_data, json_score_only, window)

    if json_score_only == 1:
        suf_name = reaching_const.JSON_SCORE_FILE
    elif json_score_only == 2:
        suf_name = reaching_const.JSON_SCORE_FILE_2
    else:
        suf_name = reaching_const.JSON_COMBINED_FILE

    if output_json_folder is None:
        output_json_folder = os.path.join(reaching_const.INPUT_FOLDER, "json_files")
    # print('output_json_folder', output_json_folder)
    with open(os.path.join(output_json_folder, reaching_const.PREFIX + suf_name), 'w') as fp:
        # print(scores_frames)
        json.dump(scores_frames, fp)

    # print(data)


if __name__ == "__main__":
    poses_for_frames(json_score_only=0)
