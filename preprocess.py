import os
import itertools
import shutil
import settings

# Optional preprocessing steps useful for automation
# This section of code will take all data in subdirectories of settings.FOLDER
# and put that data in the settings.FOLDER directory. Subdirectories are then 
# deleted. All files are renamed with underscores instead of spaces.

# flatten structure
def move(destination):
    all_files = []
    for root, _dirs, files in itertools.islice(os.walk(destination), 1, None):
        for filename in files:
            all_files.append(os.path.join(root, filename))
    for filename in all_files:
        shutil.move(filename, destination)
        
move(setting.FOLDER)

# remove all subdirectories
for folder in os.listdir(settings.FOLDER):
    folder_path = os.path.join(settings.FOLDER, folder)
    if os.path.isdir(folder_path):
        os.rmdir(folder_path)

# rename all files in settings.FOLDER to not have spaces
for vid in os.listdir(settings.FOLDER):
    vid_path = os.path.join(settings.FOLDER, vid)
    print(vid_path.replace(" ","_"))
    os.rename(vid_path, vid_path.replace(" ","_"))