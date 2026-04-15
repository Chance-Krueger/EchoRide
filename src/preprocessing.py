from pathlib import Path
from scipy.io import wavfile
import soundfile as sf
import librosa
import numpy as np

from visualization import plot_waveform, plot_spectrogram, plot_mel_spectrogram




# Noise reduction, filtering, normalization

# Read one .wav file and return the actual waveform plus its sample rate.
def load_audio_file(file_path):
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    if file_path.suffix.lower() != ".wav":
        raise ValueError(f"Expected a .wav file, got: {file_path}")

    sample_rate, audio = wavfile.read(file_path)
    return audio, sample_rate


# Look at one loaded clip and understand what dealing with.
def inspect_audio_properties(audio, sample_rate, file_path):
    num_samples = len(audio)
    duration = num_samples / sample_rate

    if audio.ndim == 1:
        channels = 1
    else:
        channels = audio.shape[1]

    info = {
        "sample_rate": sample_rate,
        "num_samples": num_samples,
        "duration": duration,
        "channels": channels,
        "dtype": str(audio.dtype),
    }

    return info

# Make every clip use the same sample rate.
def resample_audio(audio, original_sr, target_sr):
    if original_sr == target_sr:
        return audio, original_sr
    
    audio = audio.astype(np.float32)

    resampled_audio = librosa.resample(
        y=audio,
        orig_sr=original_sr,
        target_sr=target_sr
    )

    return resampled_audio, target_sr



# Make sure every clip has one channel.
def convert_to_mono(audio):
    if audio.ndim == 1:
        return audio
    return audio.mean(axis=1)


# Remove unnecessary quiet sections at the beginning and end.
def trim_silence(audio, threshold):
    # Ensure mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Absolute amplitude
    abs_audio = np.abs(audio)

    # Find indices where audio is above threshold
    above_thresh = np.where(abs_audio > threshold)[0]

    # If nothing is above threshold, return original
    if len(above_thresh) == 0:
        return audio

    start = above_thresh[0]
    end = above_thresh[-1] + 1  # include last sample

    # trimmed_audio
    return audio[start:end]


# Put all clips on a similar amplitude scale.
def normalize_audio(audio):
    # Convert to float so division behaves correctly
    audio = audio.astype(np.float32)

    max_val = np.max(np.abs(audio))

    if max_val == 0:
        return audio

    # normalized_audio
    return audio / max_val


# Force every clip to have the same duration.
def pad_or_crop_audio(audio, sample_rate, target_duration):
    target_length = int(target_duration * sample_rate)
    current_length = len(audio)

    # If shorter → pad with zeros at the end
    if current_length < target_length:
        amount_to_pad = target_length - current_length
        return np.pad(audio, (0, amount_to_pad), mode='constant') # padded_audio

    # If longer → crop to target length
    if current_length > target_length:
        return audio[:target_length] # cropped_audio

    # Already correct length
    return audio


# Apply the full standardization pipeline to one clip.
def preprocess_audio(file_path, target_sr, target_duration, silence_threshold=500):

    # 1. Load
    audio, sample_rate = load_audio_file(file_path)

    # 2. Convert to mono
    audio = convert_to_mono(audio)

    # 3. Trim silence BEFORE resampling (threshold is based on original amplitude)
    audio = trim_silence(audio, threshold=silence_threshold)

    # 4. Resample
    audio, sample_rate = resample_audio(audio, sample_rate, target_sr)

    # 5. Normalize to [-1, 1]
    audio = normalize_audio(audio)

    # 6. Pad or crop to fixed duration
    audio = pad_or_crop_audio(audio, sample_rate, target_duration)

    print("After trim_silence:", len(audio), "samples")

    return audio, sample_rate



# Apply preprocessing to every file in dataset index.
def preprocess_dataset(dataset, target_sr, target_duration, silence_threshold=500):
    processed_dataset = []

    for entry in dataset:
        file_path = entry["file_path"]
        label = entry["label"]

        processed_audio, processed_sr = preprocess_audio(
            file_path=file_path,
            target_sr=target_sr,
            target_duration=target_duration,
            silence_threshold=silence_threshold
        )

        processed_entry = {
            "file_path": file_path,
            "label": label,
            "audio": processed_audio,
            "sample_rate": processed_sr
        }

        processed_dataset.append(processed_entry)

    return processed_dataset



def main():
    file_path = "/Users/chancekrueger/Documents/GitHub/EchoRide/data/raw/FrontPass/FrontPass_L2R_HeavyWind.wav"

    processed_audio, sr = preprocess_audio(
        file_path=file_path,
        target_sr=16000,
        target_duration=2.0,
        silence_threshold=500
    )

    raw_audio, raw_sr = load_audio_file(file_path)

    print("Raw audio length:", len(raw_audio))


    print("=== FINAL OUTPUT ===")
    print("Sample rate:", sr)
    print("Num samples:", len(processed_audio))
    print("Duration:", len(processed_audio) / sr)
    print("Max amplitude:", np.max(np.abs(processed_audio)))

    # NEW: Charts
    plot_waveform(processed_audio, sr, "Processed Waveform")
    plot_spectrogram(processed_audio, sr, "Processed Audio Spectrogram")
    plot_mel_spectrogram(processed_audio, sr, "Processed Mel Spectrogram")





    dataset = [
        {"file_path": "data/raw/FrontPass/FrontPass_L2R_HeavyWind.wav", "label": "FrontPass"},
        {"file_path": "data/raw/FrontPass/FrontPass_L2R_NC_Engine.wav", "label": "FrontPass"},
    ]

    processed = preprocess_dataset(
        dataset,
        target_sr=16000,
        target_duration=2.0
    )

    print("\n=== DATASET RESULTS ===")
    for entry in processed:
        print(entry["file_path"], len(entry["audio"]), entry["sample_rate"])

main()


if __name__ == "__main__":
    main()
