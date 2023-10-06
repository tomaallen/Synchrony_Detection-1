## preprocessing
import os
import itertools
import shutil

# Optional preprocessing steps useful for automation
# This section of code will take all data in subdirectories of PATH and put that
# data in the PATH directory.Subdirectories are then deleted. All files
PATH = "D:\\test"

# flatten structure
def move(destination):
    all_files = []
    for root, _dirs, files in itertools.islice(os.walk(destination), 1, None):
        for filename in files:
            all_files.append(os.path.join(root, filename))
    for filename in all_files:
        shutil.move(filename, destination)
        
move(PATH)

# remove all subdirectories
for folder in os.listdir(PATH):
    folder_path = os.path.join(PATH, folder)
    if os.path.isdir(folder_path):
        os.rmdir(folder_path)

# rename all files in PATH to not have spaces
for vid in os.listdir(PATH):
    vid_path = os.path.join(PATH, vid)
    print(vid_path.replace(" ","_"))
    os.rename(vid_path, vid_path.replace(" ","_"))