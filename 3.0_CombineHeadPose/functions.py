import numpy as np

def filter_dict(initial_dict, conf_threshold):
    # This function filters the dictionary by removing the frames where there are more than 2 ppl and where there is not mom-baby pair detected
    # It also returns whether each keypoint is above confidence threshold for both mum and baby in each frame
    
    ai_dyad, confident = [], []
 
    # Check dict
    for frame, info in initial_dict.items():
        # print('****************Analyzing frame # '+frame+' *****************************')
        # print(frame, ':', info, '\n')
        # print()
        for key, values  in info.items():

            # Check if there is a mom-baby couple in the frame
            if (key!='Count'):
                mom_is_here = 0 
                baby_is_here = 0
                # iterate in the first person's data
                for person in values:
                    for key, data  in person.items():
                        if (key == 'baby_ma_id'):
                            if (data == 0): # is mom
                                mom_is_here = 1
                                mum_conf = np.array([person['Body'][x][2] for x in person['Body']])
                            elif (data == 1): # is baby
                                baby_is_here = 1
                                baby_conf = np.array([person['Body'][x][2] for x in person['Body']])
    
                # check if we have a couple of mom-baby
                if (mom_is_here == 1 & baby_is_here == 1):
                    # print()
                    # print('This is a mom-baby couple -> Potentially a Good Frame, Check More!\n')
                    # filtered_dict.append({'Frame': frame, 'Info': values}) # Save this frame in the list: next check will be the keypoint confidence score
                    ai_dyad.append(True)
                    
                    confident.append((mum_conf > conf_threshold) & (baby_conf > conf_threshold))
                                    
                else:
                    # print()
                    # print('This is NOT a mom-baby couple -> Discard Frame!\n')
                    ai_dyad.append(False)   
                    confident.append(np.full(25, False))
                   

    return np.array(ai_dyad), np.stack(confident)
