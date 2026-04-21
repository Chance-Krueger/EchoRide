from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


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


# Print the mapping from class index to class name.
def print_label_mapping(label_encoder):
    print("\n=== LABEL MAPPING ===")
    for idx, label in enumerate(label_encoder.classes_):
        print(f"{idx}: {label}")



# Split dataset into train and test sets using stratification.
def split_dataset(X, y_encoded, test_size=0.25, random_state=42):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded
    )

    return X_train, X_test, y_train, y_test




def main():
    print("=== Building model dataset ===")

    # Step 1: Build dataset
    X, y = build_model_dataset()
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # Step 2: Encode labels
    y_encoded, label_encoder = encode_labels(y)
    print("Classes:", label_encoder.classes_)

    # Step 3: Split dataset
    X_train, X_test, y_train, y_test = split_dataset(X, y_encoded)

    print("\n=== SPLIT RESULTS ===")
    print("Train X:", X_train.shape)
    print("Train y:", y_train.shape)
    print("Test X:", X_test.shape)
    print("Test y:", y_test.shape)

    # Step 4: Class balance check
    print("\nTrain class counts:", {c: list(y_train).count(c) for c in set(y_train)})
    print("Test class counts:", {c: list(y_test).count(c) for c in set(y_test)})

    # Step 5: Sanity check sample
    print("\nSample train label:", y_train[0])
    print("Sample test label:", y_test[0])

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()