from pathlib import Path

### Handles loading audio files or streaming audio

# This function decides where raw audio lives.
def get_raw_data_path():

    raw_path = Path("")

    if raw_path.is_dir():
        return raw_path
    else:
        raise f"The path {raw_path} does not exist in the directory"

    pass

# This function finds the class names from folder structure.
def get_category_folders():
    pass

# Given one category folder, find all .wav files in it.
def get_wav_files_in_category():
    pass

# This turns a category folder into a label.
def extract_label_from_folder():
    pass

# Main function in file that combines all the smaller functions and creates the dataset list.
def build_file_index():
    pass

# Before loading actual audio, it is really useful to print a basic summary.
def summarize_dataset():
    pass
