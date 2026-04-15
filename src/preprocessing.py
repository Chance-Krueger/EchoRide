from scipy.io import wavfile
import soundfile as sf
import librosa
import numpy as np



# Noise reduction, filtering, normalization

# Read one .wav file and return the actual waveform plus its sample rate.
def load_audio_file(file_path):
    sample_rate, audio = wavfile.read(file_path)
    return audio, sample_rate


# Look at one loaded clip and understand what dealing with.
def inspect_audio_properties(audio, sample_rate, file_path):
    num_samples = len(audio)
    duration = num_samples / sample_rate
    info = sf.info(file_path)

    return {
        "sample_rate": sample_rate,
        "num_samples": num_samples,
        "duration": duration,
        "channels": info.channels
    }

# Make every clip use the same sample rate.
def resample_audio(audio, original_sr, target_sr, file_path):
    if original_sr == target_sr:
        return audio, original_sr
    
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

    trimmed_audio = audio[start:end]
    return trimmed_audio


# Put all clips on a similar amplitude scale.
def normalize_audio(audio):
    # Convert to float so division behaves correctly
    audio = audio.astype(np.float32)

    max_val = np.max(np.abs(audio))

    if max_val == 0:
        return audio

    normalized_audio = audio / max_val
    return normalized_audio


# Force every clip to have the same duration.
def pad_or_crop_audio(audio, sample_rate, target_duration):
    target_length = int(target_duration * sample_rate)
    current_length = len(audio)

    # If shorter → pad with zeros at the end
    if current_length < target_length:
        amount_to_pad = target_length - current_length
        padded_audio = np.pad(audio, (0, amount_to_pad), mode='constant')
        return padded_audio

    # If longer → crop to target length
    if current_length > target_length:
        cropped_audio = audio[:target_length]
        return cropped_audio

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
    audio, sample_rate = resample_audio(audio, sample_rate, target_sr, file_path)

    # 5. Normalize to [-1, 1]
    audio = normalize_audio(audio)

    # 6. Pad or crop to fixed duration
    audio = pad_or_crop_audio(audio, sample_rate, target_duration)

    return audio, sample_rate



# Apply preprocessing to every file in dataset index.
def preprocess_dataset():
    pass


def main():
    file_path = "/Users/chancekrueger/Documents/GitHub/EchoRide/data/raw/FrontPass/FrontPass_L2R_HeavyWind.wav"

    # Load audio
    audio, sample_rate = load_audio_file(file_path)

    print("=== ORIGINAL ===")
    print(inspect_audio_properties(audio, sample_rate, file_path))
    print("Shape before mono:", audio.shape)

    # Convert to mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    print("\n=== AFTER MONO ===")
    print("Num samples:", len(audio))

    # Trim silence
    trimmed = trim_silence(audio, threshold=500)

    print("\n=== AFTER TRIM SILENCE ===")
    print("Num samples:", len(trimmed))
    print("Duration:", len(trimmed) / sample_rate)

    # Resample
    target_sr = 16000
    resampled_audio, new_sr = resample_audio(trimmed, sample_rate, target_sr, file_path)

    print("\n=== AFTER RESAMPLING ===")
    print({
        "sample_rate": new_sr,
        "num_samples": len(resampled_audio),
        "duration": len(resampled_audio) / new_sr
    })

    normalized = normalize_audio(resampled_audio)

    print("\n=== AFTER NORMALIZATION ===")
    print("Max amplitude:", np.max(np.abs(normalized)))
    print("Min amplitude:", np.min(normalized))
    print("dtype:", normalized.dtype)

    # Pad or crop to 2 seconds for testing
    target_duration = 2.0
    final_audio = pad_or_crop_audio(normalized, new_sr, target_duration)

    print("\n=== AFTER PAD/CROP ===")
    print("Num samples:", len(final_audio))
    print("Duration:", len(final_audio) / new_sr)


main()