% MDRQA Trials 
% RESULTS is a double-variable holding the following recurrence variables:
%    1.  Size of the RP
%    2.  %REC  - percentage of recurrent points
%    3.  %DET  - percentage of diagonally adjacent recurrent points
%    4.  MeanL - average length of diagonal recurrent points
%    5.  MaxL  - maximum length of diagonally adjacent recurrent points
%    6.  EntrL - Shannon entropy of distribution of diagonal lines
%    7.  %LAM  - percentage of vertically adjacent recurrent points
%    8.  MeanV - average length of vertically adjacent recurrent points
%    9.  MaxV  - maximum length of vertically adjacent recurrent points
%    10. EntrV - Shannon entropy of distribution of vertical lines

clear
clc
close all

participant_list = {}; %initializing the participant list to add in the IDs manually

%% Loading pose data from excel files and storing in structure

addpath 'X:\LEAP_PCI\Singapore_PCI\Pose_RQA\MRQA_Conf0.3_BR_best' %path for data
current_filepath = 'X:\LEAP_PCI\Singapore_PCI\Pose_RQA\MRQA_Conf0.3_BR_best\';

data_size = length(participant_list);

pose_coordinates_all(data_size) = struct; %holds all the pose coordinates raw data

for i = 1:length(participant_list)

    current_file = dir(strcat(current_filepath,'*',participant_list(i),'*NeckNose.xlsx')); %this is to read all the files in the folder irrespective of Cam used, as long as it ends with NeckNose
    current_filename = current_file.name;
    disp(current_filename);

    pose_coordinates_current = table2array(readtable(current_filename));     
    pose_coordinates_all(i).ID = participant_list(i);
    pose_coordinates_all(i).data = pose_coordinates_current;

    % adding this because for some weird reason, some of the excel files
    % are actually cells with numbers written as strings in them instead of
    % them being integers in a table.. :(
    if(iscell(pose_coordinates_current)==0)
        pose_coordinates_current(any(isnan(pose_coordinates_current),2),:) = [];
        pose_coordinates_all(i).data_withoutnans = pose_coordinates_current;
    else
        pose_coordinates_current = str2double(pose_coordinates_current);
        pose_coordinates_current(any(isnan(pose_coordinates_current),2),:) = [];
        pose_coordinates_all(i).data_withoutnans = pose_coordinates_current;
    end

end

%% Smoothing pose data using different techniques, but sgolay works best so far

data_size = length(pose_coordinates_all);
window_size = 75; %data is smoothened every 75 frames

for i = 1:data_size
%     pose_coordinates_all(i).smooth_data_movmean = smoothdata(pose_coordinates_all(i).data, "movmean",window_size,"omitnan");
%     pose_coordinates_all(i).smooth_data_gaussian = smoothdata(pose_coordinates_all(i).data, "gaussian",window_size,"omitnan");
    pose_coordinates_all(i).smooth_data_sgolay = smoothdata(pose_coordinates_all(i).data, "sgolay",window_size,"omitnan");
    
end

%% Calculating velocities of neck and nose (mum and baby)

step_size = 2; % every 2 frames instantaneous velocity, calculated over 80ms here

pose_velocities_all(length(pose_coordinates_all)) = struct; %storing all the raw velocities data, calculated from the smoothened pose data

for i = 1:length(pose_coordinates_all)
    temp_data = pose_coordinates_all(i).smooth_data_sgolay;
%     if(iscell(temp_data)) 
%         temp_data = str2double(temp_data); 
%     end
    k = 1;
    for j = 1:step_size:length(temp_data)-1

        if (isnan(temp_data(j,1)) || isnan(temp_data(j+1,1)))
            pose_velocities_all(i).neck_velocity_baby(k) = NaN;
            pose_velocities_all(i).nose_velocity_baby(k) = NaN; 
            pose_velocities_all(i).neck_velocity_mum(k) = NaN; 
            pose_velocities_all(i).nose_velocity_mum(k) = NaN; 
        else
            % calculating velocities: sqrt((x2-x1)^2 + (y2-y1)^2)/(t2-t1);
            pose_velocities_all(i).neck_velocity_baby(k) = sqrt((temp_data(j+1,1) - temp_data(j,1))^2 + (temp_data(j+1,2) - temp_data(j,2))^2)/1000; %pixel coorindates per ms?
            pose_velocities_all(i).nose_velocity_baby(k) = sqrt((temp_data(j+1,3) - temp_data(j,3))^2 + (temp_data(j+1,4) - temp_data(j,4))^2)/1000;
            pose_velocities_all(i).neck_velocity_mum(k) = sqrt((temp_data(j+1,5) - temp_data(j,5))^2 + (temp_data(j+1,6) - temp_data(j,6))^2)/1000;
            pose_velocities_all(i).nose_velocity_mum(k) = sqrt((temp_data(j+1,7) - temp_data(j,7))^2 + (temp_data(j+1,8) - temp_data(j,8))^2)/1000;
        end

        k = k+1;
    end
end



%% %%%% POSE %%%% : getting the best possible delay and embedding paramaters for the subjects

close all
addpath 'X:\LEAP_PCI\Singapore_PCI\Pose_RQA\mdembedding-master\mdembedding-master' % embedding estimation parameters path

pose_RQA_parameters(data_size) = struct; %holds all the RQA parameters for pose

for i = 1:length(pose_coordinates_all)
     
     pose_RQA_parameters(i).ID = pose_coordinates_all(i).ID;
   
     max_lag = 100;  % change based on validity of plots produced!
     max_embed = 10; % change based on validity of plots produced!

     dyadic_pose_data = pose_coordinates_all(i).smooth_data_sgolay;
     dyadic_pose_data(any(isnan(dyadic_pose_data),2),:) = []; %removing Nans and concatenating the data

     % delay estimation
     pose_RQA_parameters(i).delay  = mdDelay(dyadic_pose_data, 'maxLag', max_lag, 'plottype', 'all','criterion','localMin');
     % embedding dimension estimation
     [pose_RQA_parameters(i).fnnpercent,  pose_RQA_parameters(i).embeddingDimension] = mdFnn(dyadic_pose_data, round(pose_RQA_parameters(i).delay ),'maxEmb', max_embed);

end


%% %%%% POSE %%%% : choosing the embed parameter for the dyad

for i=1:length(pose_coordinates_all)

    [min_fnn, min_fnn_embed] = min(pose_RQA_parameters(i).fnnpercent);
    pose_RQA_parameters(i).embed_dyad = min(min_fnn_embed);

end

mean_embed = mean([pose_RQA_parameters.embed_dyad]);
mean_delay = mean([pose_RQA_parameters.delay]);
max_embed = max([pose_RQA_parameters.embed_dyad]);

%% %%%% POSE %%%% : RUNNING MDRQA recurrence 

addpath 'X:\LEAP_PCI\Singapore_PCI\Pose_RQA\MdRQA-master\MdRQA-master' % path to the actual mdRQA code

data_size = length(pose_coordinates_all);

pose_MDRQA_results(data_size) = struct;

for i = 1:data_size

     pose_MDRQA_results(i).ID = pose_coordinates_all(i).ID;

     embed = pose_RQA_parameters(i).embed_dyad;
     delay = pose_RQA_parameters(i).delay;

     dyadic_pose_data = pose_coordinates_all(i).smooth_data_sgolay;

     dyadic_data_withnans = dyadic_pose_data;
     dyadic_pose_data(any(isnan(dyadic_pose_data),2),:) = []; %removing Nans and concatenating the data

     pose_MDRQA_results(i).final_data_length = length(dyadic_pose_data);
     pose_MDRQA_results(i).usable_percent = 100*(length(dyadic_pose_data)/length(dyadic_data_withnans)); % calculating data usability based on %NaNs

     thresh = 0.01;

     if(pose_MDRQA_results(i).usable_percent > 35) %change usability threshold based on dataset!

     [~, pose_MDRQA_results(i).RESULTS, pose_MDRQA_results(i).PARAMETERS, pose_MDRQA_results(i).b] = MDRQA(dyadic_velocities_data,embed,delay,'euc',thresh,1);
     pose_MDRQA_results(i).recpercent = pose_MDRQA_results(i).RESULTS(2);

     % commenting out the other variables for now because not relevant
     % (since they dont't work with discontinuities in the data)
%      pose_MDRQA_results(i).diagpercent = pose_MDRQA_results(i).RESULTS(3);
%      pose_MDRQA_results(i).meanL = pose_MDRQA_results(i).RESULTS(4);
%      pose_MDRQA_results(i).maxL = pose_MDRQA_results(i).RESULTS(5);
%      pose_MDRQA_results(i).entropydiag = pose_MDRQA_results(i).RESULTS(6);
%      pose_MDRQA_results(i).lampercent = pose_MDRQA_results(i).RESULTS(7);
%      pose_MDRQA_results(i).meanV = pose_MDRQA_results(i).RESULTS(8);
%      pose_MDRQA_results(i).maxV = pose_MDRQA_results(i).RESULTS(9);
%      pose_MDRQA_results(i).entropyvert = pose_MDRQA_results(i).RESULTS(10);
% 
     else
         pose_MDRQA_results(i).recpercent = NaN;
     end

end

%% %%%% VELOCITY %%%% : getting the best possible delay and embedding paramaters for the subjects

close all
addpath 'X:\LEAP_PCI\Singapore_PCI\Pose_RQA\mdembedding-master\mdembedding-master' % embedding estimation parameters path

velocity_RQA_parameters(data_size) = struct; %holds all the RQA parameters for velocity data

for i = 1:length(pose_velocities_all)
     
     velocity_RQA_parameters(i).ID = pose_velocities_all(i).ID;
   
     max_lag = 100;  % change based on validity of plots produced!
     max_embed = 10;  % change based on validity of plots produced!

     dyadic_velocities_data = transpose([pose_velocities_all(i).neck_velocity_baby;pose_velocities_all(i).nose_velocity_baby;pose_velocities_all(i).neck_velocity_mum;pose_velocities_all(i).nose_velocity_mum]);
     dyadic_velocities_data(any(isnan(dyadic_velocities_data),2),:) = [];

     % delay estimation
     velocity_RQA_parameters(i).delay  = mdDelay(dyadic_velocities_data, 'maxLag', max_lag, 'plottype', 'all','criterion','localMin');
     % embedding dimension estimation
     [velocity_RQA_parameters(i).fnnpercent,  velocity_RQA_parameters(i).embeddingDimension] = mdFnn(dyadic_velocities_data, round(velocity_RQA_parameters(i).delay ),'maxEmb', max_embed);

end


%% %%%% VELOCITY %%%% : choosing the embed parameter for the dyad

for i=1:length(pose_velocities_all)

    [min_fnn, min_fnn_embed] = min(velocity_RQA_parameters(i).fnnpercent);
    velocity_RQA_parameters(i).embed_dyad = min(min_fnn_embed);

end

mean_embed = mean([velocity_RQA_parameters.embed_dyad]);
mean_delay = mean([velocity_RQA_parameters.delay]);
max_embed = max([velocity_RQA_parameters.embed_dyad]);

%% %%%% VELOCITY %%%% :  RUNNING MDRQA recurrence 

addpath 'X:\LEAP_PCI\Singapore_PCI\Pose_RQA\MdRQA-master\MdRQA-master' % path to the actual mdRQA code
processing_mode = 'Dyadic';

data_size = length(pose_velocities_all);

velocity_MDRQA_results(data_size) = struct;

for i = 1:data_size

     velocity_MDRQA_results(i).ID = pose_velocities_all(i).ID;

     embed = velocity_RQA_parameters(i).embed_dyad;
     delay = velocity_RQA_parameters(i).delay;

     dyadic_velocities_data = transpose([pose_velocities_all(i).neck_velocity_baby;pose_velocities_all(i).nose_velocity_baby;pose_velocities_all(i).neck_velocity_mum;pose_velocities_all(i).nose_velocity_mum]);

     dyadic_data_withnans = dyadic_velocities_data;
     dyadic_velocities_data(any(isnan(dyadic_velocities_data),2),:) = []; %removing Nans and concatenating the data

     velocity_MDRQA_results(i).final_data_length = length(dyadic_velocities_data);
     velocity_MDRQA_results(i).usable_percent = 100*(length(dyadic_velocities_data)/length(dyadic_data_withnans)); % calculating data usability based on %NaNs

     thresh = 0.01;

     if(velocity_MDRQA_results(i).usable_percent > 35)  %change usability threshold based on dataset!

     [~, velocity_MDRQA_results(i).RESULTS, velocity_MDRQA_results(i).PARAMETERS, velocity_MDRQA_results(i).b] = MDRQA(dyadic_velocities_data,embed,delay,'euc',thresh,1);
     velocity_MDRQA_results(i).recpercent = velocity_MDRQA_results(i).RESULTS(2);

     % commenting out the other variables for now because not relevant
     % (since they dont't work with discontinuities in the data)   
%      velocity_MDRQA_results(i).diagpercent = velocity_MDRQA_results(i).RESULTS(3);
%      velocity_MDRQA_results(i).meanL = velocity_MDRQA_results(i).RESULTS(4);
%      velocity_MDRQA_results(i).maxL = velocity_MDRQA_results(i).RESULTS(5);
%      velocity_MDRQA_results(i).entropydiag = velocity_MDRQA_results(i).RESULTS(6);
%      velocity_MDRQA_results(i).lampercent = velocity_MDRQA_results(i).RESULTS(7);
%      velocity_MDRQA_results(i).meanV = velocity_MDRQA_results(i).RESULTS(8);
%      velocity_MDRQA_results(i).maxV = velocity_MDRQA_results(i).RESULTS(9);
%      velocity_MDRQA_results(i).entropyvert = velocity_MDRQA_results(i).RESULTS(10);

     else
         velocity_MDRQA_results(i).recpercent = NaN;
     end

end




