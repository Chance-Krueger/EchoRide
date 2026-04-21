# Maps direction → vibration pattern



# Core mapping
VIBRATION_MAP = {
    "FrontPass": {
        "side": "front",
        "pattern": "smooth",
        "intensity": 0.5,
        "duration": 1.5
    },

    "LeftPass": {
        "side": "left",
        "pattern": "smooth",
        "intensity": 0.6,
        "duration": 1.5
    },

    "RightPass": {
        "side": "right",
        "pattern": "smooth",
        "intensity": 0.6,
        "duration": 1.5
    },

    "RearPass": {
        "side": "rear",
        "pattern": "smooth",
        "intensity": 0.6,
        "duration": 1.5
    },

    "LeftTurn": {
        "side": "left",
        "pattern": "pulse",
        "intensity": 0.8,
        "duration": 2.0
    },

    "RightTurn": {
        "side": "right",
        "pattern": "pulse",
        "intensity": 0.8,
        "duration": 2.0
    },

    "RearCrash": {
        "side": "rear",
        "pattern": "burst",
        "intensity": 1.0,
        "duration": 1.0
    }
}

# Get vibration pattern
def get_vibration_pattern(label):

    if label not in VIBRATION_MAP:
        raise ValueError(f"Unknown label: {label}")

    return VIBRATION_MAP[label]


# Simulates what the vibration would feel like (prints it). Replace this later with hardware control.
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
        print("-> Pulsing vibration (on/off rhythm)")

    elif pattern_type == "burst":
        print("-> Rapid alert bursts")

    else:
        print("-> Unknown pattern type")



def main():
    test_labels = [
        "FrontPass",
        "LeftPass",
        "RightTurn",
        "RearCrash"
    ]

    for label in test_labels:
        pattern = get_vibration_pattern(label)
        simulate_vibration(pattern)


if __name__ == "__main__":
    main()