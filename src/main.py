from direction_model import (
    get_or_build_dataset,
    get_or_train_model,
    encode_labels,
    print_label_mapping,
    split_dataset,
    summarize_split,
    evaluate_model,
    predict_one
)

from vibration import (
    get_vibration_pattern,
    simulate_vibration
)

import numpy as np


def collapse_to_front_back(y):
    return np.array([
        "Front" if label == "FrontPass" else "Back"
        for label in y
    ])


def main():
    use_cached_dataset = True
    force_rebuild_dataset = False

    use_saved_model = True
    force_retrain_model = False

    print("=== ECHORIDE PIPELINE START ===")

    print("\n=== BUILDING / LOADING DATASET ===")
    X, y = get_or_build_dataset(
        use_cached_dataset=use_cached_dataset,
        force_rebuild_dataset=force_rebuild_dataset,
        target_sr=16000,
        target_duration=2.0,
        silence_threshold=500
    )

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    y = collapse_to_front_back(y)

    print("\n=== COLLAPSED LABELS ===")
    print("Unique labels:", np.unique(y))

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
        force_retrain_model=force_retrain_model
    )

    results = evaluate_model(model, X_test, y_test, model_label_encoder)

    print("\n=== SAMPLE PREDICTION ===")
    sample_vector = X_test[0]
    true_label = model_label_encoder.inverse_transform([y_test[0]])[0]
    predicted_label = predict_one(model, sample_vector, model_label_encoder)

    print("True label:     ", true_label)
    print("Predicted label:", predicted_label)

    print("\n=== VIBRATION MAPPING ===")
    vibration_pattern = get_vibration_pattern(predicted_label)
    simulate_vibration(vibration_pattern)

    print("\n=== ECHORIDE PIPELINE COMPLETE ===")


if __name__ == "__main__":
    main()