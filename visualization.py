import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from pathlib import Path
# from preprocessing import load_audio_file, preprocess_audio

# Be able to take one audio file and visually inspect


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FOLDER = BASE_DIR / "data"
RAW_FOLDER = DATA_FOLDER / "raw"


# Show amplitude over time for one waveform.
def plot_waveform(audio, sample_rate, title="Waveform"):
    # Create time axis from 0 to duration
    duration = len(audio) / sample_rate
    time = np.linspace(0, duration, len(audio))

    # Plot
    plt.figure(figsize=(10, 4))
    plt.plot(time, audio, linewidth=1)
    plt.title(title)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()

# Show how frequency changes over time.
def plot_spectrogram(audio, sample_rate, title="Spectrogram"):
    # Compute STFT
    stft = librosa.stft(audio, n_fft=1024, hop_length=256)
    
    # Magnitude
    magnitude = np.abs(stft)
    
    # Convert to decibels
    db_scale = librosa.amplitude_to_db(magnitude, ref=np.max)

    # Plot
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        db_scale,
        sr=sample_rate,
        hop_length=256,
        x_axis="time",
        y_axis="linear",
        cmap="magma"
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title(title)
    plt.tight_layout()
    plt.show()


# Show a mel-scaled spectrogram
def plot_mel_spectrogram(audio, sample_rate, title="Mel Spectrogram"):
    # Compute mel spectrogram (power)
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=1024,
        hop_length=256,
        n_mels=128,
        power=2.0
    )

    # Convert power to decibels
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Plot
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        mel_db,
        sr=sample_rate,
        hop_length=256,
        x_axis="time",
        y_axis="mel",
        cmap="magma"
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title(title)
    plt.tight_layout()
    plt.show()