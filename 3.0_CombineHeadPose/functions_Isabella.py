# This function filters the dictionary by removing the frames where there are less than 2 ppl and where there is not mom-baby pair detected
def filter_dict(initial_dict):
    discarded_frames = 0
    check_baby_ma_id = 0
    filtered_dict = [] # empty final list of dict 
 
    # Check dict
    for frame, info in initial_dict.items():
        return_frame = 0
        print('****************Analyzing frame # '+frame+' *****************************')
        #print(frame, ':', info, '\n')
        #print()
        for key, values  in info.items():
            
            # Check how many ppl are detected per frame
            if (key=='Count'): 
                if (values >= 2):
                    print('2 or More People detected -> Potentially a Good Frame, Check More!\n')
                    check_baby_ma_id = 1   # Check for baby_ma_id          

                else:
                    print('Not enough people detected -> Discard frame! ')
                    discarded_frames = discarded_frames + 1    
                    check_baby_ma_id = 0   # No need to check for baby_ma_id -> frame already discarded

            # Check if there is a mom-baby couple in the frame
            else:
                if (check_baby_ma_id == 1):
                    mom_is_here = 0 
                    baby_is_here = 0

                    # iterate in all people's data
                    for i in range(len(values)):
                        try:
                            for key, data  in values[i].items():
                                if (key == 'baby_ma_id'):
                                    if (data == 0): # is mom
                                        mom_is_here += 1
                                        print('mom_is_here')
                                        print(mom_is_here)
                                    elif (data == 1): # is baby
                                        baby_is_here += 1  
                                        print('baby_is_here') 
                                        print(baby_is_here)

                        except:
                            mom_is_here = 0
                            baby_is_here = 0

                    # check if we have a couple of mom-baby
                    if (mom_is_here == 1 & baby_is_here == 1):
                        print()
                        print('This is a mom-baby couple -> Potentially a Good Frame, Check More!\n')
                        return_frame = 1
                    
                    else:
                        print()
                        print('This is NOT a mom-baby couple -> Discard Frame!\n')
                        discarded_frames = discarded_frames + 1    

                else:    
                    break    

        if (return_frame == 1): # Save this frame in the list: next check will be the keypoint confidence score
            d1 = {'Frame': frame, 'Info': values}
            filtered_dict.append(d1)
  
    print('discarded_frames: '+ str(discarded_frames))

    return filtered_dict, discarded_frames


# This function filters the dictionary by discharing the hip and neck keypoints with a confidence level <.3
def filter_conf(filtered_dict, discarded_frames):

    final_dict = []
    print('...............Filtering keypoints...................')    
    for i in range(len(filtered_dict)):
        print()
        # print(filtered_dict[i])
        for key, values  in filtered_dict[i].items():
            if key == ('Info'):
                # print(values)
                bad_frame = 0 # flag to keep track if one person involved in the frame has conf score <.3
                for k in range(len(values)): # Iterates 2 times -> 2 ppl per frame
                    if ((values[k]['baby_ma_id'] == 0) or (values[k]['baby_ma_id'] == 1)):
                        if ((values[k]['Body']['Neck'][2] < 0.3) or (values[k]['Body']['Nose'][2] < 0.3)): # Checks Neck and MidHip keypoints confidence score
                            bad_frame = bad_frame + 1 # the person analysed has conf score <.3

                print(i)
                if (bad_frame == 0): # both ppl have conf scores >0.3 
                    final_dict.append(filtered_dict[i])
                    print('\nThis is a GOOD frame!')
                else: # one or both ppl have conf score <.3 
                    print('\nDiscard this frame!')
                    discarded_frames = discarded_frames +1
                                   
    return final_dict, discarded_frames








# This function filters the dictionary by removing the frames where there are less than 2 ppl and where there is not mom-baby pair detected
def filter_dict_second(initial_dict, worksheet, row):
    discarded_frames = 0
    check_baby_ma_id = 0
    filtered_dict = [] # empty final list of dict 
 
    

    # Check dict
    for frame, info in initial_dict.items():
        return_frame = 0
        print('****************Analyzing frame # '+frame+' *****************************')
        #print(frame, ':', info, '\n')
        #print()
        for key, values  in info.items():
            
            # Check how many ppl are detected per frame
            if (key=='Count'): 
                if (values >= 2):
                    print('2 or More People detected -> Potentially a Good Frame, Check More!\n')
                    check_baby_ma_id = 1   # Check for baby_ma_id          

                else:
                    print('Not enough people detected -> Discard frame! ')
                    discarded_frames = discarded_frames + 1    
                    check_baby_ma_id = 0   # No need to check for baby_ma_id -> frame already discarded
                    # Add NaN to excel
                    row += 1 
                    worksheet.write(row, 0, "NaN")
                    worksheet.write(row, 1, "NaN")
                    worksheet.write(row, 2, "NaN")
                    worksheet.write(row, 3, "NaN")
                    worksheet.write(row, 4, "NaN")
                    worksheet.write(row, 5, "NaN")
                    worksheet.write(row, 6, "NaN")
                    worksheet.write(row, 7, "NaN")


            # Check if there is a mom-baby couple in the frame
            else:
                if (check_baby_ma_id == 1):
                    mom_is_here = 0 
                    baby_is_here = 0

                    # iterate in all people's data
                    for i in range(len(values)):
                        try:
                            for key, data  in values[i].items():
                                if (key == 'baby_ma_id'):
                                    if (data == 0): # is mom
                                        mom_is_here += 1
                                        print('mom_is_here')
                                        print(mom_is_here)
                                    elif (data == 1): # is baby
                                        baby_is_here += 1  
                                        print('baby_is_here') 
                                        print(baby_is_here)


                        except:
                            mom_is_here = 0
                            baby_is_here = 0



                    # OLD iterate in the first person's data
                    #for key, data  in values[0].items():
                    #    if (key == 'baby_ma_id'):
                    #        if (data == 0): # is mom
                    #            mom_is_here = 1
                    #        elif (data == 1): # is baby
                    #            baby_is_here = 1   
                    
                    # iterate in the second person's data
                    #try:
                    #    for key, data  in values[1].items():
                    #        if (key == 'baby_ma_id'):
                    #            if (data == 0): # is mom
                    #                mom_is_here = 1
                    #            elif (data == 1): # is baby
                    #                baby_is_here = 1  
                    #except:
                    #    mom_is_here = 0
                    #    baby_is_here = 0

                    #else:
                    #    for key, data  in values[1].items():
                    #        if (key == 'baby_ma_id'):
                    #            if (data == 0): # is mom
                    #                mom_is_here = 1
                    #            elif (data == 1): # is baby
                    #                baby_is_here = 1




                    # check if we have a couple of mom-baby
                    if (mom_is_here == 1 & baby_is_here == 1):
                        print()
                        print('This is a mom-baby couple -> Potentially a Good Frame, Check More!\n')
                        return_frame = 1
                    
                    else:
                        print()
                        print('This is NOT a mom-baby couple -> Discard Frame!\n')
                        discarded_frames = discarded_frames + 1    
                        # Add NaN to excel
                        row += 1 
                        worksheet.write(row, 0, "NaN")
                        worksheet.write(row, 1, "NaN")
                        worksheet.write(row, 2, "NaN")
                        worksheet.write(row, 3, "NaN")
                        worksheet.write(row, 4, "NaN")
                        worksheet.write(row, 5, "NaN")
                        worksheet.write(row, 6, "NaN")
                        worksheet.write(row, 7, "NaN")

                else:    
                    break    

        if (return_frame == 1): # Save this frame in the list: next check will be the keypoint confidence score
            d1 = {'Frame': frame, 'Info': values}
            filtered_dict.append(d1)

            
            
    print('discarded_frames: '+ str(discarded_frames))
    

    return filtered_dict, discarded_frames, row










# This function filters the dictionary by discharing the hip and neck keypoints with a confidence level <.3
def filter_conf_second(filtered_dict, discarded_frames, worksheet, row):

    final_dict = []
    baby_kcoords = [] # contains X_Nb, Y_Nb, X_MHb, Y_MHb
    mom_kcoords = [] # contains X_Nm, Y_Nm, X_MHm, Y_MHm
    print('...............Filtering keypoints...................')    
    for i in range(len(filtered_dict)):
        for key, values  in filtered_dict[i].items():
            if key == ('Info'):
                bad_frame = 0 # flag to keep track if one person involved in the frame has conf score <.3
                for k in range(len(values)): # Iterates n times -> n ppl per frame
                    if ((values[k]['baby_ma_id'] == 0) or (values[k]['baby_ma_id'] == 1)):
                        if ((values[k]['Body']['Neck'][2] < 0.3) or (values[k]['Body']['Nose'][2] < 0.3)): # Checks Neck and MidHip keypoints confidence score
                            bad_frame = bad_frame + 1 # the person analysed has conf score <.3
                    
                    # add Neck and MidHip coords to mom keypoints coords list   
                    if ((values[k]['baby_ma_id'] == 0)): # mom
                        mom_kcoords.append(values[k]['Body']['Neck'][0])
                        mom_kcoords.append(values[k]['Body']['Neck'][1])          
                        mom_kcoords.append(values[k]['Body']['Nose'][0])
                        mom_kcoords.append(values[k]['Body']['Nose'][1])               

                    
                    # add Neck and MidHip coords to baby keypoints coords list 
                    if ((values[k]['baby_ma_id'] == 1)): # baby
                        baby_kcoords.append(values[k]['Body']['Neck'][0])
                        baby_kcoords.append(values[k]['Body']['Neck'][1])
                        baby_kcoords.append(values[k]['Body']['Nose'][0])
                        baby_kcoords.append(values[k]['Body']['Nose'][1]) 


                if (bad_frame == 0): # both ppl have conf scores >0.3 
                    final_dict.append(filtered_dict[i])
                    print('\nThis is a GOOD frame!')
                    # Fill excel spreadsheet 
                    row += 1 # new row of the excel file
                    worksheet.write(row, 0, baby_kcoords[0])
                    worksheet.write(row, 1, baby_kcoords[1])
                    worksheet.write(row, 2, baby_kcoords[2])
                    worksheet.write(row, 3, baby_kcoords[3])
                    worksheet.write(row, 4, mom_kcoords[0])
                    worksheet.write(row, 5, mom_kcoords[1])
                    worksheet.write(row, 6, mom_kcoords[2])
                    worksheet.write(row, 7, mom_kcoords[3])

                else: # one or both ppl have conf score <.3 
                    print('\nDiscard this frame!')
                    discarded_frames = discarded_frames +1
                    # add NAN to excel
                    row += 1 
                    worksheet.write(row, 0, "NaN")
                    worksheet.write(row, 1, "NaN")
                    worksheet.write(row, 2, "NaN")
                    worksheet.write(row, 3, "NaN")
                    worksheet.write(row, 4, "NaN")
                    worksheet.write(row, 5, "NaN")
                    worksheet.write(row, 6, "NaN")
                    worksheet.write(row, 7, "NaN")



            #else:
            #    baby_kcoords = [] # contains X_Nb, Y_Nb, X_MHb, Y_MHb
            #    mom_kcoords = [] # contains X_Nm, Y_Nm, X_MHm, Y_MHm


    return final_dict, discarded_frames, row




def read_NaNs(workbook):
    wrkbk = openpyxl.load_workbook(workbook) 
    sh = wrkbk.active 
        
    # Iterate through excel and check if dataset is usable  
    consecutive_NaNs = 0
    total_consecutive_NaNs = []
    for i in range(1, sh.max_row+1): 
        for j in range(1, 2): # Only first coloumn since they are all the same for the NaNs
            cell_obj = sh.cell(row=i, column=j) 
            if (i > 2):
                previous_cell_obj = sh.cell(row=i-1, column=j) 
                if ((cell_obj.value == 'NaN') & (previous_cell_obj.value == 'NaN')):
                    consecutive_NaNs += 1
                    if (consecutive_NaNs > 750):  # Set here how many seconds are acceptable
                        print('Dataset NOT usable')
                        return
                else:
                    consecutive_NaNs = 0


    
    


