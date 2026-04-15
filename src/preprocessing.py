from scipy.io import wavfile
import soundfile as sf
import librosa



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
def trim_silence():
    pass


# Put all clips on a similar amplitude scale.
def normalize_audio():
    pass


# Force every clip to have the same duration.
def pad_or_crop_audio():
    pass

# Apply the full standardization pipeline to one clip.


def preprocess_audio():
    pass


# Apply preprocessing to every file in dataset index.
def preprocess_dataset():
    pass


def main():
    file_path = "/Users/chancekrueger/Documents/GitHub/EchoRide/data/raw/FrontPass/FrontPass_L2R_HeavyWind.wav"

    # Load audio
    audio, sample_rate = load_audio_file(file_path)

    print("=== BEFORE RESAMPLING ===")
    print(inspect_audio_properties(audio, sample_rate, file_path))

    # Choose a target sample rate
    target_sr = 16000

    # Resample
    audio = convert_to_mono(audio)
    resampled_audio, new_sr = resample_audio(audio, sample_rate, target_sr, file_path)


    print("\n=== AFTER RESAMPLING ===")
    print({
        "sample_rate": new_sr,
        "num_samples": len(resampled_audio),
        "duration": len(resampled_audio) / new_sr
    })

main()