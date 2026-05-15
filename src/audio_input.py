from pathlib import Path
from collections import Counter

from label_mapping import map_label


# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FOLDER = BASE_DIR / "data"


# Raw data location
def get_raw_data_path():
    raw_path = DATA_FOLDER / "raw"

    if raw_path.is_dir():
        return raw_path

    raise FileNotFoundError(f"The path {raw_path} does not exist in the directory")



# Folder / file discovery
def get_category_folders(raw_data_path):
    category_folders = []

    for path in raw_data_path.iterdir():
        if path.is_dir():
            category_folders.append(path)

    return sorted(category_folders)


def get_wav_files_in_category(category_path):
    wav_files = []

    for file in Path(category_path).iterdir():
        if file.is_file() and file.suffix.lower() == ".wav":
            wav_files.append(file)

    return sorted(wav_files)


def extract_label_from_folder(category_path):
    return Path(category_path).name



# Build dataset entries from folder structure
def build_file_index(raw_data_path, use_mapped_labels=True, allowed_labels=None):
    
    dataset = []

    category_folders = get_category_folders(raw_data_path)

    for category_folder in category_folders:
        original_label = extract_label_from_folder(category_folder)

        if use_mapped_labels:
            final_label = map_label(original_label)
        else:
            final_label = original_label

        if allowed_labels is not None and final_label not in allowed_labels:
            continue

        wav_files = get_wav_files_in_category(category_folder)

        for wav_file in wav_files:
            entry = {
                "file_path": wav_file,
                "original_label": original_label,
                "label": final_label
            }
            dataset.append(entry)

    return dataset



# Summary helpers
def summarize_dataset(dataset):
    print(f"Total files: {len(dataset)}")

    if not dataset:
        print("Dataset is empty.")
        return

    original_counts = Counter(entry["original_label"] for entry in dataset)
    final_counts = Counter(entry["label"] for entry in dataset)

    print("\nOriginal label counts:")
    for label, count in sorted(original_counts.items()):
        print(f"{label}: {count}")

    print("\nFinal label counts:")
    for label, count in sorted(final_counts.items()):
        print(f"{label}: {count}")



def main():
    data_dir = get_raw_data_path()

    dataset = build_file_index(
        raw_data_path=data_dir,
        use_mapped_labels=True
    )

    summarize_dataset(dataset)

    if dataset:
        print("\nExample entry:")
        print(dataset[0])


if __name__ == "__main__":
    main()