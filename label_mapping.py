import numpy as np



# Core label mapping
LABEL_MAP = {
    "FrontPass": "Front",

    "RearPass": "Back",
    "RearCrash": "Back",

    "LeftPass": "Left",
    "LeftTurn": "Left",

    "RightPass": "Right",
    "RightTurn": "Right"
}


# Map a single label
def map_label(label):
    if label not in LABEL_MAP:
        raise ValueError(f"Unknown label: {label}")

    return LABEL_MAP[label]


# Map a list/array of labels
def map_labels(y):
    return np.array([map_label(label) for label in y])



# Get list of final class names
def get_class_names():
    return sorted(list(set(LABEL_MAP.values())))



# Print mapping (for debugging)
def print_label_mapping():
    print("\n=== LABEL MAPPING (ORIGINAL → FINAL) ===")
    for original, new in LABEL_MAP.items():
        print(f"{original} → {new}")

    print("\nFinal classes:", get_class_names())




def main():
    test_labels = [
        "FrontPass",
        "LeftPass",
        "RightTurn",
        "RearCrash"
    ]

    print_label_mapping()

    mapped = map_labels(test_labels)

    print("\nOriginal:", test_labels)
    print("Mapped:  ", mapped)


if __name__ == "__main__":
    main()