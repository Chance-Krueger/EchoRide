import numpy as np
import librosa

from preprocessing import load_audio_file, preprocess_audio # TESTING


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
def extract_features_from_audio(audio, sample_rate):
    # Individual feature groups
    mfcc_features = extract_mfcc_features(audio, sample_rate)                     # (26,)
    rms_features = extract_rms_feature(audio)                                     # (2,)
    centroid_features = extract_spectral_centroid_feature(audio, sample_rate)     # (2,)
    zcr_features = extract_zero_crossing_feature(audio)                           # (2,)
    bandwidth_features = extract_spectral_bandwidth_feature(audio, sample_rate)   # (2,)
    rolloff_features = extract_spectral_rolloff_feature(audio, sample_rate)       # (2,)

    # Concatenate all features into one vector
    full_vector = np.concatenate([
        mfcc_features,
        rms_features,
        centroid_features,
        zcr_features,
        bandwidth_features,
        rolloff_features
    ]).astype(np.float32)

    return full_vector

# Take the processed dataset and convert it into model-ready data
def extract_features_from_dataset():
    pass


def main():
    dataset = [
        {"file_path": "data/raw/FrontPass/FrontPass_L2R_HeavyWind.wav",
            "label": "FrontPass"},
        {"file_path": "data/raw/FrontPass/FrontPass_L2R_NC_Engine.wav",
            "label": "FrontPass"},
        {"file_path": "data/raw/FrontPass/FrontPass_R2L_HeavyWind.wav",
            "label": "FrontPass"},
        {"file_path": "data/raw/FrontPass/FrontPass_R2L_NC_Engine.wav",
            "label": "FrontPass"},
    ]

    print("=== PROCESSING DATASET ===")

    for entry in dataset:
        file_path = entry["file_path"]
        label = entry["label"]

        print(f"\n--- File: {file_path} ---")

        # Load raw audio
        raw_audio, raw_sr = load_audio_file(file_path)
        print("Raw length:", len(raw_audio), "Raw SR:", raw_sr)

        # Preprocess
        processed_audio, processed_sr = preprocess_audio(
            file_path=file_path,
            target_sr=16000,
            target_duration=2.0,
            silence_threshold=500
        )
        print("Processed length:", len(processed_audio),
              "Processed SR:", processed_sr)

        # Extract MFCC features
        mfcc_vec = extract_mfcc_features(
            processed_audio, processed_sr, n_mfcc=13)
        print("MFCC shape:", mfcc_vec.shape)
        print("MFCC sample:", mfcc_vec[:5], "...")

        # Optional: visualize
        # visualize_audio_sample(file_path)

    processed_audio, sr = preprocess_audio(
        file_path="/Users/chancekrueger/Documents/GitHub/EchoRide/data/raw/FrontPass/FrontPass_L2R_HeavyWind.wav",
        target_sr=16000,
        target_duration=2.0,
        silence_threshold=500
    )

    rms_feat = extract_rms_feature(processed_audio, sr)

    print("RMS feature vector:", rms_feat)
    print("Shape:", rms_feat.shape)

    processed_audio, sr = preprocess_audio(
        file_path="/Users/chancekrueger/Documents/GitHub/EchoRide/data/raw/FrontPass/FrontPass_L2R_HeavyWind.wav",
        target_sr=16000,
        target_duration=2.0,
        silence_threshold=500
    )

    centroid_feat = extract_spectral_centroid_feature(processed_audio, sr)

    print("Spectral centroid feature:", centroid_feat)
    print("Shape:", centroid_feat.shape)

    zcr_feat = extract_zero_crossing_feature(processed_audio)
    print("ZCR feature vector:", zcr_feat)
    print("Shape:", zcr_feat.shape)

    bandwidth_feat = extract_spectral_bandwidth_feature(processed_audio, sr)
    print("Spectral bandwidth feature:", bandwidth_feat)
    print("Shape:", bandwidth_feat.shape)

    rolloff_feat = extract_spectral_rolloff_feature(processed_audio, sr)
    print("Spectral rolloff feature:", rolloff_feat)
    print("Shape:", rolloff_feat.shape)

    processed_audio, sr = preprocess_audio(
    file_path="data/raw/FrontPass/FrontPass_L2R_HeavyWind.wav",
    target_sr=16000,
    target_duration=2.0,
    silence_threshold=500
)

    vec = extract_features_from_audio(processed_audio, sr)

    print("Full feature vector shape:", vec.shape)
    print("Sample:", vec[:10])




    

    print("\n=== DONE ===")


main()
