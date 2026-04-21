import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


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


# Train a baseline Random Forest classifier.
def train_random_forest(X_train, y_train, random_state=42):

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=random_state,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)
    return model


# Evaluate the trained model and print: accuracy, classification report, confusion matrix
def evaluate_model(model, X_test, y_test, label_encoder):
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_,
        zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred)

    print("\n=== EVALUATION ===")
    print(f"Accuracy: {accuracy:.4f}\n")

    print("Classification Report:")
    print(report)

    print("Confusion Matrix:")
    print(cm)

    return {
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": cm,
        "y_pred": y_pred
    }

# Predict one sample from a single feature vector.
def predict_one(model, feature_vector, label_encoder):

    feature_vector = np.asarray(feature_vector, dtype=np.float32)

    if feature_vector.ndim == 1:
        feature_vector = feature_vector.reshape(1, -1)

    pred_encoded = model.predict(feature_vector)[0]
    pred_label = label_encoder.inverse_transform([pred_encoded])[0]

    return pred_label



def main():
    import numpy as np
from audio_input import get_raw_data_path, build_file_index
from preprocessing import preprocess_dataset
from feature_extract import extract_features_from_dataset

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# -----------------------------
# Label Encoding
# -----------------------------
def encode_labels(y):
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    return y_encoded, label_encoder


# -----------------------------
# Train/Test Split
# -----------------------------
def split_dataset(X, y_encoded, test_size=0.25, random_state=42):
    return train_test_split(
        X,
        y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded
    )


# -----------------------------
# Random Forest Training
# -----------------------------
def train_random_forest(X_train, y_train, random_state=42):
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=random_state,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)
    return model


# -----------------------------
# Model Evaluation
# -----------------------------
def evaluate_model(model, X_test, y_test, label_encoder):
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_,
        zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred)

    print("\n=== EVALUATION ===")
    print(f"Accuracy: {accuracy:.4f}\n")

    print("Classification Report:")
    print(report)

    print("Confusion Matrix:")
    print(cm)

    return {
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": cm,
        "y_pred": y_pred
    }


# -----------------------------
# Predict One Sample
# -----------------------------
def predict_one(model, feature_vector, label_encoder):
    feature_vector = np.asarray(feature_vector, dtype=np.float32)

    if feature_vector.ndim == 1:
        feature_vector = feature_vector.reshape(1, -1)

    pred_encoded = model.predict(feature_vector)[0]
    pred_label = label_encoder.inverse_transform([pred_encoded])[0]

    return pred_label


# -----------------------------
# Dataset Builder
# -----------------------------
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


# -----------------------------
# MAIN
# -----------------------------
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
    print("Test X:", X_test.shape)

    # Step 4: Train model
    print("\n=== TRAINING RANDOM FOREST ===")
    model = train_random_forest(X_train, y_train)

    # Step 5: Evaluate model
    evaluate_model(model, X_test, y_test, label_encoder)

    # Step 6: Predict one sample (use first test sample)
    print("\n=== SINGLE SAMPLE PREDICTION ===")
    sample_vector = X_test[0]
    true_label = label_encoder.inverse_transform([y_test[0]])[0]
    pred_label = predict_one(model, sample_vector, label_encoder)

    print("True Label:     ", true_label)
    print("Predicted Label:", pred_label)

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()