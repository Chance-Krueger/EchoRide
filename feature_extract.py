import numpy as np

from audio_input import get_raw_data_path, build_file_index
from preprocessing import preprocess_dataset
from wav2vec_features import extract_wav2vec_features


# Add Wav2Vec feature vector to each processed dataset entry.
def build_feature_dataset(processed_dataset):

    feature_dataset = []

    for entry in processed_dataset:
        audio = entry["audio"]
        sample_rate = entry["sample_rate"]
        label = entry["label"]
        file_path = entry["file_path"]

        feature_vector = extract_wav2vec_features(audio, sample_rate)

        feature_entry = {
            "file_path": file_path,
            "original_label": entry.get("original_label"),
            "label": label,
            "sample_rate": sample_rate,
            "features": feature_vector
        }

        feature_dataset.append(feature_entry)

    return feature_dataset


# Convert processed audio dataset into model-ready X and y.
def extract_features_from_dataset(processed_dataset):

    feature_dataset = build_feature_dataset(processed_dataset)

    X = []
    y = []

    for entry in feature_dataset:
        X.append(entry["features"])
        y.append(entry["label"])

    X = np.array(X, dtype=np.float32)
    y = np.array(y)

    if X.ndim != 2:
        raise ValueError(f"X should be 2D, got shape {X.shape}")

    if y.ndim != 1:
        raise ValueError(f"y should be 1D, got shape {y.shape}")

    if len(X) != len(y):
        raise ValueError(f"X/y mismatch: len(X)={len(X)}, len(y)={len(y)}")

    return X, y


def summarize_feature_dataset(feature_dataset):
    print("\n=== FEATURE DATASET SUMMARY ===")
    print(f"Total samples: {len(feature_dataset)}")

    if not feature_dataset:
        print("Feature dataset is empty.")
        return

    print(f"Feature length: {len(feature_dataset[0]['features'])}")

    label_counts = {}
    for entry in feature_dataset:
        label = entry["label"]
        label_counts[label] = label_counts.get(label, 0) + 1

    print("\nLabel counts:")
    for label, count in sorted(label_counts.items()):
        print(f"{label}: {count}")


def main():
    raw_data_path = get_raw_data_path()

    raw_dataset = build_file_index(
        raw_data_path=raw_data_path,
        use_mapped_labels=True
    )

    processed_dataset = preprocess_dataset(
        raw_dataset,
        target_sr=16000,
        target_duration=2.0,
        trim=True,
        top_db=30,
        normalize=True,
        force_mono=True
    )

    feature_dataset = build_feature_dataset(processed_dataset)
    summarize_feature_dataset(feature_dataset)

    X, y = extract_features_from_dataset(processed_dataset)

    print("\n=== MODEL-READY DATA ===")
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    if len(X) > 0:
        print("First feature vector length:", len(X[0]))
        print("First label:", y[0])


if __name__ == "__main__":
    main()