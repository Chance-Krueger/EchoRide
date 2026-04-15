import numpy as np
import matplotlib.pyplot as plt

# Be able to take one audio file and visually inspect

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
def plot_spectrogram():
    pass


# Show a mel-scaled spectrogram
def plot_mel_spectrogram():
    pass


# Visually compare raw and processed audio.
def compare_waveforms():
    pass


# Visually compare raw and processed frequency content.
def compare_spectrograms():
    pass


# main pipeline helper for a single file.
def visualize_audio_sample():
    pass