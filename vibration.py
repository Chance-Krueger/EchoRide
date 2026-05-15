VIBRATION_MAP = {
    "Front": {
        "side": "front",
        "pattern": "smooth",
        "intensity": 0.5,
        "duration": 1.5
    },

    "Left": {
        "side": "left",
        "pattern": "pulse",
        "intensity": 0.8,
        "duration": 2.0
    },

    "Right": {
        "side": "right",
        "pattern": "pulse",
        "intensity": 0.8,
        "duration": 2.0
    },

    "Back": {
        "side": "rear",
        "pattern": "burst",
        "intensity": 1.0,
        "duration": 1.0
    }
}


def get_vibration_pattern(label):
    if label not in VIBRATION_MAP:
        raise ValueError(f"Unknown label: {label}")

    return VIBRATION_MAP[label]


def simulate_vibration(pattern):
    side = pattern["side"]
    pattern_type = pattern["pattern"]
    intensity = pattern["intensity"]
    duration = pattern["duration"]

    print("\n=== VIBRATION OUTPUT ===")
    print(f"Side: {side}")
    print(f"Pattern: {pattern_type}")
    print(f"Intensity: {intensity}")
    print(f"Duration: {duration}")

    if pattern_type == "smooth":
        print("-> Continuous vibration")

    elif pattern_type == "pulse":
        print("-> Pulsing vibration")

    elif pattern_type == "burst":
        print("-> Rapid alert bursts")

    else:
        print("-> Unknown vibration pattern")


def main():
    test_labels = ["Front", "Left", "Right", "Back"]

    for label in test_labels:
        pattern = get_vibration_pattern(label)
        simulate_vibration(pattern)


if __name__ == "__main__":
    main()