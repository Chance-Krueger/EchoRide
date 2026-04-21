import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


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


# Train a baseline SVM classifier.
def train_svm(X_train, y_train, random_state=42):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        class_weight="balanced",
        random_state=random_state
    )

    model.fit(X_train_scaled, y_train)

    return model, scaler


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


# Evaluate the trained SVM model and print: accuracy, classification report, confusion matrix
def evaluate_scaled_model(model, scaler, X_test, y_test, label_encoder):
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)

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



# Dataset summary
def summarize_split(X_train, X_test, y_train, y_test, label_encoder):
    print("\n=== DATA SPLIT SUMMARY ===")
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)

    print("\nTrain label counts:")
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    for label_id, count in zip(unique_train, counts_train):
        print(f"  {label_encoder.inverse_transform([label_id])[0]}: {count}")

    print("\nTest label counts:")
    unique_test, counts_test = np.unique(y_test, return_counts=True)
    for label_id, count in zip(unique_test, counts_test):
        print(f"  {label_encoder.inverse_transform([label_id])[0]}: {count}")


def main():
    print("=== BUILDING DATASET ===")
    X, y = build_model_dataset(
        target_sr=16000,
        target_duration=2.0,
        silence_threshold=500
    )

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    y_encoded, label_encoder = encode_labels(y)
    print_label_mapping(label_encoder)

    X_train, X_test, y_train, y_test = split_dataset(
        X,
        y_encoded,
        test_size=0.25,
        random_state=42
    )

    summarize_split(X_train, X_test, y_train, y_test, label_encoder)

    # print("\n=== TRAINING FOREST MODEL ===")
    # model = train_random_forest(X_train, y_train, random_state=42)

    # results = evaluate_model(model, X_test, y_test, label_encoder)

    print("\n=== TRAINING SVM MODEL ===")
    model, scaler = train_svm(X_train, y_train, random_state=42)

    results = evaluate_scaled_model(model, scaler, X_test, y_test, label_encoder)

    # Example: inspect the first test prediction
    first_pred = label_encoder.inverse_transform([model.predict(X_test[:1])[0]])[0]
    first_true = label_encoder.inverse_transform([y_test[0]])[0]

    print("\n=== SAMPLE PREDICTION ===")
    print("True label:", first_true)
    print("Predicted label:", first_pred)

    return model, label_encoder, results


if __name__ == "__main__":
    main()