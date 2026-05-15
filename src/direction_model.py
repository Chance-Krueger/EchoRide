from pathlib import Path
import joblib
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler
)

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from audio_input import (
    get_raw_data_path,
    build_file_index
)

from preprocessing import preprocess_dataset
from feature_extract import extract_features_from_dataset


BASE_DIR = Path(__file__).resolve().parent.parent

MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

DATASET_CACHE_PATH = (
    MODELS_DIR / "wav2vec_direction_dataset.npz"
)

MODEL_BUNDLE_PATH = (
    MODELS_DIR / "dnn_direction_model.joblib"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


class DirectionDNN(nn.Module):

    def __init__(self, input_dim, num_classes):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.network(x)


def build_model_dataset(
    target_sr=16000,
    target_duration=2.0
):

    raw_data_path = get_raw_data_path()

    raw_dataset = build_file_index(
        raw_data_path=raw_data_path,
        use_mapped_labels=True
    )

    processed_dataset = preprocess_dataset(
        raw_dataset,
        target_sr=target_sr,
        target_duration=target_duration,
        force_mono=False,
        augment=True
    )

    X, y = extract_features_from_dataset(
        processed_dataset
    )

    return X, y


def save_dataset_arrays(X, y):

    np.savez(
        DATASET_CACHE_PATH,
        X=X,
        y=y
    )

    print(
        f"Saved dataset arrays to: "
        f"{DATASET_CACHE_PATH}"
    )


def load_dataset_arrays():

    data = np.load(
        DATASET_CACHE_PATH,
        allow_pickle=True
    )

    X = data["X"]
    y = data["y"]

    print(
        f"Loaded dataset arrays from: "
        f"{DATASET_CACHE_PATH}"
    )

    return X, y


def get_or_build_dataset(
    use_cached_dataset=True,
    force_rebuild_dataset=False,
    target_sr=16000,
    target_duration=2.0
):

    if (
        use_cached_dataset
        and not force_rebuild_dataset
        and DATASET_CACHE_PATH.exists()
    ):

        return load_dataset_arrays()

    print(
        "Building dataset from raw audio "
        "using Wav2Vec features..."
    )

    X, y = build_model_dataset(
        target_sr=target_sr,
        target_duration=target_duration
    )

    save_dataset_arrays(X, y)

    return X, y


def encode_labels(y):

    label_encoder = LabelEncoder()

    y_encoded = label_encoder.fit_transform(y)

    return y_encoded, label_encoder


def print_label_mapping(label_encoder):

    print("\n=== LABEL MAPPING ===")

    for idx, label in enumerate(
        label_encoder.classes_
    ):

        print(f"{idx}: {label}")


def split_dataset(
    X,
    y_encoded,
    test_size=0.10,
    random_state=42
):

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    return train_test_split(
        X_scaled,
        y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded
    )


def summarize_split(
    X_train,
    X_test,
    y_train,
    y_test,
    label_encoder
):

    print("\n=== DATA SPLIT SUMMARY ===")

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)

    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)

    print("\nTrain label counts:")

    unique_train, counts_train = np.unique(
        y_train,
        return_counts=True
    )

    for label_id, count in zip(
        unique_train,
        counts_train
    ):

        label = label_encoder.inverse_transform(
            [label_id]
        )[0]

        print(f"{label}: {count}")

    print("\nTest label counts:")

    unique_test, counts_test = np.unique(
        y_test,
        return_counts=True
    )

    for label_id, count in zip(
        unique_test,
        counts_test
    ):

        label = label_encoder.inverse_transform(
            [label_id]
        )[0]

        print(f"{label}: {count}")


def train_dnn(
    X_train,
    y_train,
    input_dim,
    num_classes,
    epochs=200,
    learning_rate=0.0003
):

    X_train_tensor = torch.tensor(
        X_train,
        dtype=torch.float32
    ).to(DEVICE)

    y_train_tensor = torch.tensor(
        y_train,
        dtype=torch.long
    ).to(DEVICE)

    model = DirectionDNN(
        input_dim=input_dim,
        num_classes=num_classes
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    print("\n=== TRAINING DNN ===")
    print("Device:", DEVICE)

    for epoch in range(epochs):

        model.train()

        optimizer.zero_grad()

        outputs = model(X_train_tensor)

        loss = criterion(
            outputs,
            y_train_tensor
        )

        loss.backward()

        optimizer.step()

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"| Loss: {loss.item():.4f}"
        )

    return model


def evaluate_model(
    model,
    X_test,
    y_test,
    label_encoder
):

    model.eval()

    X_test_tensor = torch.tensor(
        X_test,
        dtype=torch.float32
    ).to(DEVICE)

    with torch.no_grad():

        outputs = model(X_test_tensor)

        y_pred = torch.argmax(
            outputs,
            dim=1
        ).cpu().numpy()

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    report = classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_,
        zero_division=0
    )

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    print("\n=== EVALUATION ===")

    print(f"Accuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(report)

    print("Confusion Matrix:")
    print(cm)

    return {
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": cm,
        "y_pred": y_pred
    }


def predict_one(
    model,
    feature_vector,
    label_encoder
):

    model.eval()

    feature_vector = np.asarray(
        feature_vector,
        dtype=np.float32
    )

    if feature_vector.ndim == 1:
        feature_vector = feature_vector.reshape(1, -1)

    feature_tensor = torch.tensor(
        feature_vector,
        dtype=torch.float32
    ).to(DEVICE)

    with torch.no_grad():

        output = model(feature_tensor)

        pred_encoded = torch.argmax(
            output,
            dim=1
        ).cpu().numpy()[0]

    pred_label = label_encoder.inverse_transform(
        [pred_encoded]
    )[0]

    return pred_label


def get_or_train_model(
    X_train,
    y_train,
    label_encoder,
    use_saved_model=False,
    force_retrain_model=True,
    epochs=200,
    learning_rate=0.0003
):

    input_dim = X_train.shape[1]

    num_classes = len(
        label_encoder.classes_
    )

    model = train_dnn(
        X_train=X_train,
        y_train=y_train,
        input_dim=input_dim,
        num_classes=num_classes,
        epochs=epochs,
        learning_rate=learning_rate
    )

    return model, label_encoder