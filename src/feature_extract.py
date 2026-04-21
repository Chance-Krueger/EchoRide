import numpy as np
import librosa

from audio_input import get_raw_data_path, build_file_index
from preprocessing import preprocess_dataset


# Extracts features needed for direction detection


# Compute MFCCs for one audio clip and summarize them.
def extract_mfcc_features(audio, sample_rate, n_mfcc=13):
    # Compute MFCC matrix: shape (n_mfcc, time_frames)
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=n_mfcc,
        n_fft=1024,
        hop_length=256
    )

    # Mean across time axis → shape (n_mfcc,)
    mfcc_mean = np.mean(mfcc, axis=1)

    # Std across time axis → shape (n_mfcc,)
    mfcc_std = np.std(mfcc, axis=1)

    # Concatenate into one feature vector → shape (2 * n_mfcc,)
    features = np.concatenate([mfcc_mean, mfcc_std])

    return features


# Compute RMS energy and summarize it
def extract_rms_feature(audio, sample_rate=16000):
    # Compute RMS over frames
    rms = librosa.feature.rms(
        y=audio,
        frame_length=1024,
        hop_length=256
    )[0]  # shape: (time_frames,)

    # Mean and std across time
    rms_mean = np.mean(rms)
    rms_std = np.std(rms)

    return np.array([rms_mean, rms_std], dtype=np.float32)

# Summarize where the energy sits in the frequency spectrum.
def extract_spectral_centroid_feature(audio, sample_rate):
    # Compute spectral centroid → shape (1, time_frames)
    centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=sample_rate,
        n_fft=1024,
        hop_length=256
    )[0]  # flatten to (time_frames,)

    # Mean and std across time
    centroid_mean = np.mean(centroid)
    centroid_std = np.std(centroid)

    return np.array([centroid_mean, centroid_std], dtype=np.float32)

# Measure noisiness / signal roughness
def extract_zero_crossing_feature(audio):
    # Compute zero-crossing rate → shape (1, time_frames)
    zcr = librosa.feature.zero_crossing_rate(
        y=audio,
        frame_length=1024,
        hop_length=256
    )[0]  # flatten to (time_frames,)

    # Mean and std across time
    zcr_mean = np.mean(zcr)
    zcr_std = np.std(zcr)

    return np.array([zcr_mean, zcr_std], dtype=np.float32)

# Summarize the spread of frequencies
def extract_spectral_bandwidth_feature(audio, sample_rate):
    # Compute spectral bandwidth → shape (1, time_frames)
    bandwidth = librosa.feature.spectral_bandwidth(
        y=audio,
        sr=sample_rate,
        n_fft=1024,
        hop_length=256
    )[0]  # flatten to (time_frames,)

    # Mean and std across time
    bandwidth_mean = np.mean(bandwidth)
    bandwidth_std = np.std(bandwidth)

    return np.array([bandwidth_mean, bandwidth_std], dtype=np.float32)

# Summarize the upper-end frequency boundary of most energy
def extract_spectral_rolloff_feature(audio, sample_rate):
    # Compute spectral rolloff → shape (1, time_frames)
    rolloff = librosa.feature.spectral_rolloff(
        y=audio,
        sr=sample_rate,
        roll_percent=0.85,   # standard 85% rolloff
        n_fft=1024,
        hop_length=256
    )[0]  # flatten to (time_frames,)

    # Mean and std across time
    rolloff_mean = np.mean(rolloff)
    rolloff_std = np.std(rolloff)

    return np.array([rolloff_mean, rolloff_std], dtype=np.float32)

# main feature extractor
def extract_features_from_audio(audio, sample_rate, n_mfcc=13):
    # Individual feature groups
    audio = np.asarray(audio, dtype=np.float32).flatten()

    mfcc_features = extract_mfcc_features(audio, sample_rate, n_mfcc=n_mfcc)
    rms_features = extract_rms_feature(audio)
    centroid_features = extract_spectral_centroid_feature(audio, sample_rate)
    zcr_features = extract_zero_crossing_feature(audio)
    bandwidth_features = extract_spectral_bandwidth_feature(audio, sample_rate)
    rolloff_features = extract_spectral_rolloff_feature(audio, sample_rate)

    feature_vector = np.concatenate([
        mfcc_features,
        rms_features,
        centroid_features,
        zcr_features,
        bandwidth_features,
        rolloff_features
    ]).astype(np.float32)

    expected_length = (2 * n_mfcc) + 10  # 26 + 10 = 36 when n_mfcc=13
    if feature_vector.shape[0] != expected_length:
        raise ValueError(
            f"Feature vector length mismatch. "
            f"Expected {expected_length}, got {feature_vector.shape[0]}"
        )

    if np.isnan(feature_vector).any() or np.isinf(feature_vector).any():
        raise ValueError("Feature vector contains NaN or Inf values.")

    return feature_vector

# Take the processed dataset and convert it into model-ready data
def extract_features_from_dataset(processed_dataset, n_mfcc=13):
    feature_dataset = build_feature_dataset(processed_dataset, n_mfcc=n_mfcc)

    X = []
    y = []

    for entry in feature_dataset:
        X.append(entry["features"])
        y.append(entry["label"])

    X = np.array(X, dtype=np.float32)
    y = np.array(y)

    if len(X) != len(y):
        raise ValueError(
            f"Sample count mismatch: len(X)={len(X)} vs len(y)={len(y)}"
        )

    if X.ndim != 2:
        raise ValueError(f"X should be 2D, got shape {X.shape}")

    if y.ndim != 1:
        raise ValueError(f"y should be 1D, got shape {y.shape}")

    return X, y


# Adds a fixed-length feature vector to each processed dataset entry.
def build_feature_dataset(processed_dataset, n_mfcc=13):

    feature_dataset = []

    for entry in processed_dataset:
        audio = entry["audio"]
        sample_rate = entry["sample_rate"]
        label = entry["label"]
        file_path = entry["file_path"]

        feature_vector = extract_features_from_audio(
            audio,
            sample_rate,
            n_mfcc=n_mfcc
        )

        feature_entry = {
            "file_path": file_path,
            "label": label,
            "sample_rate": sample_rate,
            "features": feature_vector
        }

        feature_dataset.append(feature_entry)

    return feature_dataset

# Helps summarize data
def summarize_feature_dataset(feature_dataset):
    print(f"Total samples: {len(feature_dataset)}")

    if not feature_dataset:
        print("Feature dataset is empty.")
        return

    feature_length = len(feature_dataset[0]["features"])
    print(f"Feature length per sample: {feature_length}")

    label_counts = {}
    for entry in feature_dataset:
        label = entry["label"]
        label_counts[label] = label_counts.get(label, 0) + 1

    print("Label counts:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")


# Take a 1D time-varying feature and return: [mean, std, min, max]
def summarize_feature_series(series):
    series = np.asarray(series, dtype=np.float32).flatten()

    return np.array([
        np.mean(series),
        np.std(series),
        np.min(series),
        np.max(series)
    ], dtype=np.float32)


# Simple trend over time using a fitted line slope.
def compute_linear_slope(series):
    series = np.asarray(series, dtype=np.float32).flatten()

    if len(series) < 2:
        return np.array([0.0], dtype=np.float32)

    x = np.arange(len(series), dtype=np.float32)
    slope = np.polyfit(x, series, 1)[0]

    return np.array([slope], dtype=np.float32)



# Measures change over time: mean(second half) - mean(first half)
def split_halves_mean_difference(series):
    series = np.asarray(series, dtype=np.float32).flatten()

    if len(series) < 2:
        return np.array([0.0], dtype=np.float32)

    midpoint = len(series) // 2
    first_half = series[:midpoint]
    second_half = series[midpoint:]

    if len(first_half) == 0 or len(second_half) == 0:
        return np.array([0.0], dtype=np.float32)

    diff = np.mean(second_half) - np.mean(first_half)
    return np.array([diff], dtype=np.float32)



def main():
    raw_data_path = get_raw_data_path()
    raw_dataset = build_file_index(raw_data_path)

    print("=== RAW DATASET ===")
    print(f"Total indexed files: {len(raw_dataset)}")

    processed_dataset = preprocess_dataset(
        raw_dataset,
        target_sr=16000,
        target_duration=2.0,
        silence_threshold=500
    )

    feature_dataset = build_feature_dataset(processed_dataset, n_mfcc=13)
    summarize_feature_dataset(feature_dataset)

    X, y = extract_features_from_dataset(processed_dataset, n_mfcc=13)

    print("\n=== MODEL-READY DATA ===")
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    if len(X) > 0:
        print("First feature vector length:", len(X[0]))
        print("First label:", y[0])


if __name__ == "__main__":
    main()
