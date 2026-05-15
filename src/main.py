from direction_model import (
    get_or_build_dataset,
    encode_labels,
    split_dataset,
    train_dnn
)

from sklearn.metrics import accuracy_score
import torch


def main():

    print("=== ECHORIDE WAV2VEC + DNN PIPELINE START ===")

    X, y = get_or_build_dataset(
        use_cached_dataset=False,
        force_rebuild_dataset=True,
        target_sr=16000,
        target_duration=2.0
    )

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    y_encoded, label_encoder = encode_labels(y)

    X_train, X_test, y_train, y_test = split_dataset(
        X,
        y_encoded,
        test_size=0.10,
        random_state=42
    )

    model = train_dnn(
        X_train=X_train,
        y_train=y_train,
        input_dim=X_train.shape[1],
        num_classes=len(label_encoder.classes_),
        epochs=200,
        learning_rate=0.0003
    )

    model.eval()

    X_test_tensor = torch.tensor(
        X_test,
        dtype=torch.float32
    )

    with torch.no_grad():

        outputs = model(X_test_tensor)

        predictions = torch.argmax(
            outputs,
            dim=1
        ).numpy()

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print("\n=== FINAL RESULTS ===")
    print("Accuracy:", accuracy)


if __name__ == "__main__":
    main()