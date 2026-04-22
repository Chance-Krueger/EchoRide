from pathlib import Path
import joblib
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from audio_input import get_raw_data_path, build_file_index
from preprocessing import preprocess_dataset
from feature_extract import extract_features_from_dataset


# ---------------------------------
# Paths for saved artifacts
# ---------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

DATASET_CACHE_PATH = MODELS_DIR / "direction_dataset.npz"
MODEL_BUNDLE_PATH = MODELS_DIR / "forest_direction_model.joblib"


# ---------------------------------
# Dataset preparation
# ---------------------------------

def build_model_dataset(target_sr=16000, target_duration=2.0, silence_threshold=500):
    """
    Full pipeline:
        1. Index raw files
        2. Preprocess audio
        3. Extract features
        4. Return X and y
    """
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


def save_dataset_arrays(X, y, filename=DATASET_CACHE_PATH):
    """
    Save model-ready dataset arrays.
    """
    np.savez(filename, X=X, y=y)
    print(f"Saved dataset arrays to: {filename}")


def load_dataset_arrays(filename=DATASET_CACHE_PATH):
    """
    Load cached model-ready dataset arrays.
    """
    if not Path(filename).exists():
        raise FileNotFoundError(f"No saved dataset found at: {filename}")

    data = np.load(filename, allow_pickle=True)
    X = data["X"]
    y = data["y"]

    print(f"Loaded dataset arrays from: {filename}")
    return X, y


def get_or_build_dataset(
    use_cached_dataset=True,
    force_rebuild_dataset=True,
    target_sr=16000,
    target_duration=2.0,
    silence_threshold=500
):
    """
    Either load saved X/y or rebuild them from raw audio.
    """
    if use_cached_dataset and not force_rebuild_dataset and DATASET_CACHE_PATH.exists():
        return load_dataset_arrays(DATASET_CACHE_PATH)

    print("Building dataset from raw audio...")
    X, y = build_model_dataset(
        target_sr=target_sr,
        target_duration=target_duration,
        silence_threshold=silence_threshold
    )

    save_dataset_arrays(X, y, DATASET_CACHE_PATH)
    return X, y


# ---------------------------------
# Label encoding
# ---------------------------------

def encode_labels(y):
    """
    Convert string labels into numeric class IDs.
    """
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    return y_encoded, label_encoder


def print_label_mapping(label_encoder):
    print("\n=== LABEL MAPPING ===")
    for idx, label in enumerate(label_encoder.classes_):
        print(f"{idx}: {label}")


# ---------------------------------
# Train/test split
# ---------------------------------

def split_dataset(X, y_encoded, test_size=0.25, random_state=42):
    """
    Split dataset into train and test sets using stratification.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded
    )

    return X_train, X_test, y_train, y_test


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


# ---------------------------------
# Model training
# ---------------------------------

def train_random_forest(X_train, y_train, random_state=42):
    """
    Train a baseline Random Forest classifier.
    """
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=random_state,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)
    return model


# ---------------------------------
# Model save/load
# ---------------------------------

def save_model_bundle(
    model,
    label_encoder,
    filename=MODEL_BUNDLE_PATH,
    model_name="RandomForest",
    scaler=None,
    metadata=None
):
    """
    Save trained model state.
    """
    bundle = {
        "model": model,
        "label_encoder": label_encoder,
        "scaler": scaler,
        "model_name": model_name,
        "metadata": metadata or {}
    }

    joblib.dump(bundle, filename)
    print(f"Saved model bundle to: {filename}")


def load_model_bundle(filename=MODEL_BUNDLE_PATH):
    """
    Load previously saved model state.
    """
    if not Path(filename).exists():
        raise FileNotFoundError(f"No saved model found at: {filename}")

    bundle = joblib.load(filename)
    print(f"Loaded model bundle from: {filename}")
    return bundle


# ---------------------------------
# Evaluation
# ---------------------------------

def evaluate_model(model, X_test, y_test, label_encoder):
    """
    Evaluate the trained model and print:
        - accuracy
        - classification report
        - confusion matrix
    """
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


# ---------------------------------
# Single prediction helper
# ---------------------------------

def predict_one(model, feature_vector, label_encoder):
    """
    Predict one sample from a single feature vector.
    """
    feature_vector = np.asarray(feature_vector, dtype=np.float32)

    if feature_vector.ndim == 1:
        feature_vector = feature_vector.reshape(1, -1)

    pred_encoded = model.predict(feature_vector)[0]
    pred_label = label_encoder.inverse_transform([pred_encoded])[0]

    return pred_label


# ---------------------------------
# Train or load flow
# ---------------------------------

def train_and_save_model(
    X_train,
    y_train,
    label_encoder,
    filename=MODEL_BUNDLE_PATH,
    random_state=42
):
    """
    Train a forest model and save it.
    """
    print("\n=== TRAINING FOREST MODEL ===")
    model = train_random_forest(X_train, y_train, random_state=random_state)

    metadata = {
        "num_features": X_train.shape[1],
        "num_classes": len(label_encoder.classes_)
    }

    save_model_bundle(
        model=model,
        label_encoder=label_encoder,
        filename=filename,
        model_name="RandomForest",
        scaler=None,
        metadata=metadata
    )

    return model


def get_or_train_model(
    X_train,
    y_train,
    label_encoder,
    use_saved_model=True,
    force_retrain_model=True,
    filename=MODEL_BUNDLE_PATH,
    random_state=42
):
    """
    Either load a saved model or train/save a new one.
    """
    if use_saved_model and not force_retrain_model and Path(filename).exists():
        bundle = load_model_bundle(filename)
        return bundle["model"], bundle["label_encoder"]

    model = train_and_save_model(
        X_train=X_train,
        y_train=y_train,
        label_encoder=label_encoder,
        filename=filename,
        random_state=random_state
    )

    return model, label_encoder


# ---------------------------------
# Main
# ---------------------------------

def main():
    """
    Control flags:
        use_cached_dataset:
            True  -> load saved X/y if available
            False -> rebuild from raw audio

        force_rebuild_dataset:
            True  -> ignore saved dataset and rebuild

        use_saved_model:
            True  -> load trained model if available
            False -> always train fresh

        force_retrain_model:
            True  -> ignore saved model and retrain
    """
    use_cached_dataset = True
    force_rebuild_dataset = True

    use_saved_model = True
    force_retrain_model = True

    print("=== BUILDING / LOADING DATASET ===")
    X, y = get_or_build_dataset(
        use_cached_dataset=use_cached_dataset,
        force_rebuild_dataset=force_rebuild_dataset,
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

    model, model_label_encoder = get_or_train_model(
        X_train=X_train,
        y_train=y_train,
        label_encoder=label_encoder,
        use_saved_model=use_saved_model,
        force_retrain_model=force_retrain_model,
        filename=MODEL_BUNDLE_PATH,
        random_state=42
    )

    results = evaluate_model(model, X_test, y_test, model_label_encoder)

    first_pred = model_label_encoder.inverse_transform([model.predict(X_test[:1])[0]])[0]
    first_true = model_label_encoder.inverse_transform([y_test[0]])[0]

    print("\n=== SAMPLE PREDICTION ===")
    print("True label:", first_true)
    print("Predicted label:", first_pred)

    return model, model_label_encoder, results


if __name__ == "__main__":
    main()