from sklearn.preprocessing import LabelEncoder

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

# Convert string labels into numeric class IDs.
def encode_labels(y):

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    return y_encoded, label_encoder




def main():
    print("=== Building model dataset ===")

    # Step 1: Build dataset
    X, y = build_model_dataset()

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # Step 2: Encode labels
    y_encoded, label_encoder = encode_labels(y)

    print("Encoded labels:", y_encoded)
    print("Classes:", label_encoder.classes_)

    # Step 3: Sanity check
    print("First feature vector:", X[0])
    print("Original label:", y[0])
    print("Encoded label:", y_encoded[0])

    print("=== DONE ===")


if __name__ == "__main__":
    main()