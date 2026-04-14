from scipy.io import wavfile


# Noise reduction, filtering, normalization

# Read one .wav file and return the actual waveform plus its sample rate.
def load_audio_file(file_path):
    sample_rate, audio = wavfile.read(file_path)
    return audio, sample_rate


# Look at one loaded clip and understand what dealing with.
def inspect_audio_properties():
    pass


# Make every clip use the same sample rate.
def resample_audio():
    pass


# Make sure every clip has one channel.
def convert_to_mono():
    pass

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

    print(load_audio_file("/Users/chancekrueger/Documents/GitHub/EchoRide/data/raw/FrontPass/FrontPass_L2R_HeavyWind.wav"))

main()