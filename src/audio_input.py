from pathlib import Path

### Handles loading audio files or streaming audio

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FOLDER = BASE_DIR / "data"

# This function decides where raw audio lives.
def get_raw_data_path():

    raw_path = Path(DATA_FOLDER / "raw")

    if raw_path.is_dir():
        return raw_path
    else:
        raise FileNotFoundError(f"The path {raw_path} does not exist in the directory")



# This function finds the class names from folder structure.
def get_category_folders(raw_data_path):
    category_folders = []

    for path in Path(raw_data_path).rglob("*"):
        if path.is_dir():
            category_folders.append(path)

    return category_folders

# Given one category folder, find all .wav files in it.
def get_wav_files_in_category(category_path):
    wav_files = []

    for file in Path(category_path).iterdir():
        if (Path(file).is_file()) and (Path(file).suffix == ".wav"):
            wav_files.append(file)

    return wav_files

# This turns a category folder into a label.
def extract_label_from_folder(category_path):
    return Path(category_path).name

# Main function in file that combines all the smaller functions and creates the dataset list.
def build_file_index():
    pass

# Before loading actual audio, it is really useful to print a basic summary.
def summarize_dataset():
    pass

def main():

    data_dir = get_raw_data_path()
    cat_folders = get_category_folders(data_dir)
    print(get_wav_files_in_category(cat_folders[0]))
    print()
    print(extract_label_from_folder(cat_folders[0]))


main()