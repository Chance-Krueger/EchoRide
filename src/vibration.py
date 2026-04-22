# Maps detection → vibration pattern

# Only BACK triggers vibration
VIBRATION_MAP = {
    "Back": {
        "side": "rear",
        "pattern": "pulse",
        "intensity": 0.8,
        "duration": 1.5
    },

    # Front = NO vibration (important safety decision)
    "Front": None
}


# Get vibration pattern
def get_vibration_pattern(label):
    if label not in VIBRATION_MAP:
        raise ValueError(f"Unknown label: {label}")

    return VIBRATION_MAP[label]


# Simulates vibration (replace with hardware later)
def simulate_vibration(pattern):
    if pattern is None:
        print("\n=== NO VIBRATION ===")
        print("Reason: Oncoming traffic (safe to ignore)")
        return

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
        print("-> Pulsing vibration (warning signal)")

    elif pattern_type == "burst":
        print("-> Rapid alert bursts (high urgency)")

    else:
        print("-> Unknown pattern type")


def main():
    test_labels = [
        "Front",   # should NOT vibrate
        "Back",    # should vibrate
    ]

    for label in test_labels:
        pattern = get_vibration_pattern(label)
        simulate_vibration(pattern)


if __name__ == "__main__":
    main()