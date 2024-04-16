from pathlib import Path
import os

def get_all_files_recursively_by_ext(root, ext):
    found = []
    for path in Path(root).rglob('*.{}'.format(ext)):
        found.append(str(path))
    return sorted(found)

def ensure_dir(file_path):
    directory = file_path
    if file_path[-3] == "." or file_path[-4] == ".":
        directory = os.path.dirname(file_path)
    Path(directory).mkdir(parents=True, exist_ok=True)