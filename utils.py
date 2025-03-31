import re
from pathlib import Path


def extract_index(file):
    match = re.search(r'chunk_(\d+)\.pkl', file.name)
    return int(match.group(1)) if match else float('inf')  # push non-matching files to the end

def list_chuncks_folder(path):
    folder = Path(path)

    #* collect all the files starting with *chunk_* in the input_path
    files = sorted(
        [f for f in folder.iterdir() if f.is_file() and f.name.startswith("chunk_")],
        key=extract_index)

    return files
