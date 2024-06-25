% run_mdrqa runs mdrqa analysis on position and velocity data from the
% pose synchrony pipeline. The output is position and velocity recurrence
% for each video processed. Outputs are saved in the mdrqa output folder,
% as specified in run_model2.py

function results = run_mdrqa(inputPath, outputPath, fps, USABILITY_THRESHOLD, RADIUS_THRESHOLD, MAX_LAG, MAX_EMBED)

    % dataPath is the output folder for results and contains the 
    % matlab_inputs folder
    % fps is video frames per second as an integer.
    % usability threshold, max lag and max embed are mdrqa parameters
    % which can be adjusted for the dataset
    % results is a table with the filename, percentage of data which was
    % usable from the video and recurrence metrics for position and
    % velocity data. The results table is automatically saved to the
    % outputPath at the end of this function.
    
    % test variables
    % inputPath = 'D:\KHULA\pose_synchrony\mdrqa_30fps\matlab_inputs\';
    % outputPath = 'D:\KHULA\pose_synchrony\mdrqa_30fps\';
    % fps = 30;
    % USABILITY_THRESHOLD = 35;
    % RADIUS_THRESHOLD = 0.3;
    % MAX_LAG = 100;
    % MAX_EMBED = 10;

    clc
    close all
    
    addpath 'MdRQA-master\MdRQA-master\' % path to the mdRQA code
    addpath 'mdembedding-master\mdembedding-master\' % embedding estimation parameters pat
    
    pathPattern = [inputPath, '*.csv']; % this contains best cameras only
    
    % Initialize a cell array of all file paths to analyse
    dirStruct = dir(pathPattern);
    filenames_list = {dirStruct(:).name}; % actually a cell array
    
    % SavGol smoothing parameters
    window_size = fps * 3; % data is smoothed every x frames

    % Velocity step size
    step_size = 2; % every other frame instantaneous velocity
    
    
    %% Loading pose data from excel files and storing in structure
    
    close all
    
    data_size = length(filenames_list);
    
    pose_coordinates_all(data_size) = struct; % holds all the pose coordinates raw data
    pose_velocities_all(length(pose_coordinates_all)) = struct; % storing all the raw velocities data, calculated from the smoothened pose data
    pose_RQA_parameters(data_size) = struct; %holds all the RQA parameters for pose
    
    for i = 1:data_size
        
    % correct filenaming
        current_filename = strcat(inputPath, filenames_list{i});
        disp(current_filename);
    
        pose_coordinates_current = table2array(readtable(current_filename));
        pose_coordinates_all(i).ID = filenames_list{i};
        pose_coordinates_all(i).data = pose_coordinates_current;
        
        % TODO: check if this is an issue still
        % adding this because for some weird reason, some of the excel files
        % are actually cells with numbers written as strings in them instead of
        % them being integers in a table.. :(
    
        % if(iscell(pose_coordinates_current)==0)
        %     pose_coordinates_current(any(isnan(pose_coordinates_current),2),:) = [];
        %     pose_coordinates_all(i).data_withoutnans = pose_coordinates_current;
        % else
        %     pose_coordinates_current = str2double(pose_coordinates_current);
        %     pose_coordinates_current(any(isnan(pose_coordinates_current),2),:) = [];
        %     pose_coordinates_all(i).data_withoutnans = pose_coordinates_current;
        % end

    
    % Smoothing pose data using different techniques, but sgolay works best so far
        % pose_coordinates_all(i).smooth_data_movmean = smoothdata(pose_coordinates_all(i).data, "movmean",window_size,"omitnan");
        % pose_coordinates_all(i).smooth_data_gaussian = smoothdata(pose_coordinates_all(i).data, "gaussian",window_size,"omitnan");
        if(iscell(pose_coordinates_all(i).data)) 
            pose_coordinates_all(i).data = str2double(pose_coordinates_all(i).data); 
        end
        pose_coordinates_all(i).smooth_data_sgolay = smoothdata(pose_coordinates_all(i).data, "sgolay",window_size,"omitnan");
        
    
    % Calculating velocities of neck and nose (mum and baby)
        temp_data = pose_coordinates_all(i).smooth_data_sgolay;
    
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
    
    
    
    % %%%% POSE %%%% : getting the best possible delay and embedding paramaters for the subjects             
         pose_RQA_parameters(i).ID = pose_coordinates_all(i).ID;
    
         dyadic_pose_data = pose_coordinates_all(i).smooth_data_sgolay;
         dyadic_pose_data(any(isnan(dyadic_pose_data),2),:) = []; % removing Nans and concatenating the data
    
         % delay estimation
         pose_RQA_parameters(i).delay  = mdDelay(dyadic_pose_data, 'maxLag', MAX_LAG, 'plottype', 'all','criterion','localMin');
    
         % embedding dimension estimation
         [pose_RQA_parameters(i).fnnpercent,  pose_RQA_parameters(i).embeddingDimension] = mdFnn(dyadic_pose_data, round(pose_RQA_parameters(i).delay ),'maxEmb', MAX_EMBED);
    
    
    % %%%% POSE %%%% : choosing the embed parameter for the dyad

        [min_fnn, min_fnn_embed] = min(pose_RQA_parameters(i).fnnpercent);
        pose_RQA_parameters(i).embed_dyad = min(min_fnn_embed);
    
    end
    
    mean_embed = mean([pose_RQA_parameters.embed_dyad]);
    mean_delay = mean([pose_RQA_parameters.delay]);
    maxEmbed = max([pose_RQA_parameters.embed_dyad]);
    
    
    %% %%%% POSE %%%% : RUNNING MDRQA recurrence 
    
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
        
         if(pose_MDRQA_results(i).usable_percent > 35) %change usability threshold based on dataset!
    
         % Warning: Integer operands are required for colon operator when used as index.
         warning('off', 'all');
         [~, pose_MDRQA_results(i).RESULTS, pose_MDRQA_results(i).PARAMETERS, pose_MDRQA_results(i).b] = MDRQA(dyadic_pose_data,embed,delay,'euc',RADIUS_THRESHOLD,1);
         pose_MDRQA_results(i).recpercent = pose_MDRQA_results(i).RESULTS(2);
         warning('on', 'all');
    
         else
             pose_MDRQA_results(i).recpercent = NaN;
         end
    
    end
    
    
    %% %%%% VELOCITY %%%% : getting the best possible delay and embedding paramaters for the subjects
    
    close all
    clear embed
    clear delay
    
    velocity_RQA_parameters(data_size) = struct; %holds all the RQA parameters for velocity data
    
    for i = 1:length(pose_velocities_all)
         
         velocity_RQA_parameters(i).ID = filenames_list{i};
    
         dyadic_velocities_data = transpose([pose_velocities_all(i).neck_velocity_baby;pose_velocities_all(i).nose_velocity_baby;pose_velocities_all(i).neck_velocity_mum;pose_velocities_all(i).nose_velocity_mum]);
         dyadic_velocities_data(any(isnan(dyadic_velocities_data),2),:) = [];
    
         % delay estimation
         velocity_RQA_parameters(i).delay  = mdDelay(dyadic_velocities_data, 'maxLag', MAX_LAG, 'plottype', 'all','criterion','localMin');
         % embedding dimension estimation
         [velocity_RQA_parameters(i).fnnpercent,  velocity_RQA_parameters(i).embeddingDimension] = mdFnn(dyadic_velocities_data, round(velocity_RQA_parameters(i).delay ),'maxEmb', MAX_EMBED);
    
    end
    
    
    %% %%%% VELOCITY %%%% : choosing the embed parameter for the dyad
    
    for i=1:length(pose_velocities_all)
    
        [min_fnn, min_fnn_embed] = min(velocity_RQA_parameters(i).fnnpercent);
        velocity_RQA_parameters(i).embed_dyad = min(min_fnn_embed);
    
    end
    
    mean_embed = mean([velocity_RQA_parameters.embed_dyad]);
    mean_delay = mean([velocity_RQA_parameters.delay]);
    maxEmbed = max([velocity_RQA_parameters.embed_dyad]);
    
    
    %% %%%% VELOCITY %%%% :  RUNNING MDRQA recurrence 
    
    processing_mode = 'Dyadic';
    
    velocity_MDRQA_results(data_size) = struct;
    
    for i = 1:data_size
    
         velocity_MDRQA_results(i).ID = filenames_list{i};
    
         embed = velocity_RQA_parameters(i).embed_dyad;
         delay = velocity_RQA_parameters(i).delay;
    
         dyadic_velocities_data = transpose([pose_velocities_all(i).neck_velocity_baby;pose_velocities_all(i).nose_velocity_baby;pose_velocities_all(i).neck_velocity_mum;pose_velocities_all(i).nose_velocity_mum]);
    
         dyadic_data_withnans = dyadic_velocities_data;
         dyadic_velocities_data(any(isnan(dyadic_velocities_data),2),:) = []; %removing Nans and concatenating the data
    
         velocity_MDRQA_results(i).final_data_length = length(dyadic_velocities_data);
         velocity_MDRQA_results(i).usable_percent = 100*(length(dyadic_velocities_data)/length(dyadic_data_withnans)); % calculating data usability based on %NaNs
    
         thresh = 0.05;
    
         if(velocity_MDRQA_results(i).usable_percent > USABILITY_THRESHOLD)  %change usability threshold based on dataset!
         
         % Warning: Integer operands are required for colon operator when used as index.
         warning('off', 'all');
         [~, velocity_MDRQA_results(i).RESULTS, velocity_MDRQA_results(i).PARAMETERS, velocity_MDRQA_results(i).b] = MDRQA(dyadic_velocities_data,embed,delay,'euc',thresh,1);
         warning('on', 'all');
    
         velocity_MDRQA_results(i).recpercent = velocity_MDRQA_results(i).RESULTS(2);
    
         else
             velocity_MDRQA_results(i).recpercent = NaN;
         end
    
    end
    
    
    %% %%%% POSE & VELOCITY %%%% :  Save MdRQA Results
    positionResults = struct2table(pose_MDRQA_results);
    positionResults = positionResults(:, {'ID', 'usable_percent', 'recpercent'});
    positionResults = renamevars(positionResults, {'recpercent'}, {'pos_recurrence'});
    
    velocityResults = struct2table(velocity_MDRQA_results);
    velocityResults = velocityResults(:, {'ID', 'recpercent'});
    velocityResults = renamevars(velocityResults, {'recpercent'}, {'vel_recurrence'});
    
    results = join(positionResults, velocityResults);
    writetable(results, strcat(outputPath, "\mdrqa_results.csv"));


