from direction_model import (
    build_model_dataset,
    encode_labels,
    print_label_mapping,
    split_dataset,
    summarize_split,
    train_random_forest,
    evaluate_model,
    predict_one
)

from vibration import (
    get_vibration_pattern,
    simulate_vibration
)


# Entry point: loads audio, runs algorithm, prints direction


def main():
    print("=== ECHORIDE PIPELINE START ===")

    # 1. Build dataset
    print("\n=== BUILDING DATASET ===")
    X, y = build_model_dataset(
        target_sr=16000,
        target_duration=2.0,
        silence_threshold=500
    )

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # 2. Encode labels
    y_encoded, label_encoder = encode_labels(y)
    print_label_mapping(label_encoder)

    # 3. Split dataset
    X_train, X_test, y_train, y_test = split_dataset(
        X,
        y_encoded,
        test_size=0.25,
        random_state=42
    )

    summarize_split(X_train, X_test, y_train, y_test, label_encoder)

    # 4. Train model
    print("\n=== TRAINING MODEL ===")
    model = train_random_forest(X_train, y_train, random_state=42)

    # 5. Evaluate model
    results = evaluate_model(model, X_test, y_test, label_encoder)

    # 6. Predict one sample from test set
    print("\n=== SAMPLE PREDICTION ===")
    sample_vector = X_test[0]
    true_label = label_encoder.inverse_transform([y_test[0]])[0]
    predicted_label = predict_one(model, sample_vector, label_encoder)

    print("True label:     ", true_label)
    print("Predicted label:", predicted_label)

    # 7. Map prediction to vibration pattern
    print("\n=== VIBRATION MAPPING ===")
    vibration_pattern = get_vibration_pattern(predicted_label)

    # 8. Simulate vibration output
    simulate_vibration(vibration_pattern)

    print("\n=== ECHORIDE PIPELINE COMPLETE ===")


if __name__ == "__main__":
    main()