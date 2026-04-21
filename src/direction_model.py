from audio_input import get_raw_data_path, build_file_index
from preprocessing import preprocess_dataset
from feature_extract import extract_features_from_dataset

# Core algorithm that determines direction



# Dataset preparation
def build_model_dataset(target_sr=16000, target_duration=2.0, silence_threshold=500):

    raw_data_path = get_raw_data_path()
    raw_dataset = build_file_index(raw_data_path)

    processed_dataset = preprocess_dataset(
        raw_dataset,
        target_sr=target_sr,
        target_duration=target_duration,
        silence_threshold=silence_threshold
    )

    X, y = extract_features_from_dataset(processed_dataset)

    return X, y



def main():
    print("=== Building model dataset ===")

    X, y = build_model_dataset(
        target_sr=16000,
        target_duration=2.0,
        silence_threshold=500
    )

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # Print first sample for sanity check
    print("First feature vector:", X[0])
    print("First label:", y[0])

    print("=== DONE ===")


if __name__ == "__main__":
    main()