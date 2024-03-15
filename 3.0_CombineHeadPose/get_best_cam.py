# %% imports
import pandas as pd
import synapseclient
import synapseutils
import re

# %% read data quality check file
data_quality = pd.read_csv("D:\\test\\analysis_info\\data_quality.csv")

# %% get participant and timepoint from filename (or synapse for BR)
# get Synapse metadata
manifest = pd.read_csv("C:\\Users\\bllca\OneDrive - University of Cambridge\LEAP Cambridge Shared\Pose_Synchrony\PCI_BRAINRISE\Synapse_Metadata\synapse_metadata_manifest.tsv", sep='\t')

# synapse login and get list of PCI videos
syn = synapseclient.login(authToken='eyJ0eXAiOiJKV1QiLCJraWQiOiJXN05OOldMSlQ6SjVSSzpMN1RMOlQ3TDc6M1ZYNjpKRU9VOjY0NFI6VTNJWDo1S1oyOjdaQ0s6RlBUSCIsImFsZyI6IlJTMjU2In0.eyJhY2Nlc3MiOnsic2NvcGUiOlsidmlldyIsImRvd25sb2FkIl0sIm9pZGNfY2xhaW1zIjp7fX0sInRva2VuX3R5cGUiOiJQRVJTT05BTF9BQ0NFU1NfVE9LRU4iLCJpc3MiOiJodHRwczovL3JlcG8tcHJvZC5wcm9kLnNhZ2ViYXNlLm9yZy9hdXRoL3YxIiwiYXVkIjoiMCIsIm5iZiI6MTcwMDA1OTI3NiwiaWF0IjoxNzAwMDU5Mjc2LCJqdGkiOiI0MzUyIiwic3ViIjoiMzQ2NjE2OSJ9.FjHPBA-O0GZTl7UDRXjvbL1BEaRei3OghJVaaUP6sd5zeSXF_bYRWnxU9Mi6dCI1BbK4i29UbpuQ8NE-zwnGRC9Nj0vW5i5PFebw0_-B1BZEUfFRuCjI38mFMH3B2JKob_626LoV4DhmExmHWvumbkRnAxLokCrME0G14ZuBO0oSXIT8wVDLELzjC4FibP1XUnPIsn2IUtWNUCqVycEvBg_XV7khWwsvrLl5peRDUr_f6SWnzEsiEJHe43c8KVFfuMZahpmlw9ujtMVn55y4O4Yhe67uGug4Kvp9eoWdT3Rc2M-CXF2MtTUpg2nmYLZ1Gy-CeBZ4V-IAJRMHVbj8YA') # XXX: insert authtoken

# function to rename filekeys
def filekey_rename(x: str):
    new_name = x.lower().replace(' ','_').removesuffix('.mp4').removesuffix('.mp4').removesuffix('.mov').removesuffix('.3gp')
    new_name = new_name.replace('ª','a')# remove weird a character from filename
    return  new_name

# get filenames and SynID from folder for serial download
walkedPath = synapseutils.walk_functions.walk(syn, 'syn36007626', ['file'])
fileDict = dict(list(walkedPath)[0][2])
fileDict = {filekey_rename(k): v for k, v in fileDict.items()} # edit keys to be compatible with renamed files

# %% get timepoint from manifest
def get_tp(filename: str): # requires manifest
    try: 
        synapse_ID = fileDict[filename.lower()]
    
        assessmentAge = manifest.loc[manifest.id == synapse_ID, 'assessmentAge'].iloc[0]  # assessment age in months
       
        if assessmentAge < 5:
            return 1
        if assessmentAge < 10:
            return 2
        if assessmentAge < 17:
            return 3
        if assessmentAge < 31:
            return 4
        if assessmentAge > 31:
            return 5
    except:
        return int(re.search(r'\da', filename)[0].strip('a'))

data_quality['ppt'] = data_quality.Filename.apply(lambda x : int(re.search(r'\d+', x)[0]))
data_quality['tp'] = data_quality.Filename.apply(get_tp)

# choose the file for each participant and timepoint with the most good frames
best_cams_idx = list(data_quality.groupby(['ppt', 'tp'])['Good frames'].idxmax())
best_cams = list(data_quality.iloc[best_cams_idx].Filename)

# XXX: copy best cam jsons into their own folder???