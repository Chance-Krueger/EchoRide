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
def build_file_index(raw_data_path):
    
    dataset = []

    category_folders = get_category_folders(raw_data_path)

    for category_folder in category_folders:
        label = extract_label_from_folder(category_folder)
        wav_files = get_wav_files_in_category(category_folder)

        for wav_file in wav_files:
            entry = {
                "file_path": wav_file,
                "label" : label
            }
            dataset.append(entry)

    return dataset

# Print a basic summary.
def summarize_dataset(dataset):
    
    print(len(dataset))

    # count files per label

    label_dict = {}

    for data in dataset:
        if data["label"] in label_dict:
            label_dict[data["label"]] += 1
        else:
            label_dict[data["label"]] = 1

    for key, value in label_dict.items():
        print(key, value)


def main():

    data_dir = get_raw_data_path()
    summarize_dataset(build_file_index(data_dir))


main()