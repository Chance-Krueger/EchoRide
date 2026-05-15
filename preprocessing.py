from pathlib import Path
import librosa
import numpy as np
import random


def load_audio_file(file_path, target_sr=None, mono=False):

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    audio, sample_rate = librosa.load(
        file_path,
        sr=target_sr,
        mono=mono
    )

    return audio, sample_rate


def inspect_audio_properties(audio, sample_rate):

    if audio.ndim == 1:
        channels = 1
        num_samples = len(audio)

    else:
        channels = audio.shape[0]
        num_samples = audio.shape[-1]

    duration = num_samples / sample_rate

    return {
        "sample_rate": sample_rate,
        "num_samples": num_samples,
        "duration": duration,
        "channels": channels,
        "dtype": str(audio.dtype),
    }


def convert_to_mono(audio):
    return audio


def trim_silence(audio, top_db=30):

    # Mono audio
    if audio.ndim == 1:

        trimmed_audio, _ = librosa.effects.trim(
            audio,
            top_db=top_db
        )

        return trimmed_audio

    # Stereo audio
    left_channel = audio[0]
    right_channel = audio[1]

    trimmed_left, _ = librosa.effects.trim(
        left_channel,
        top_db=top_db
    )

    trimmed_right, _ = librosa.effects.trim(
        right_channel,
        top_db=top_db
    )

    min_len = min(
        len(trimmed_left),
        len(trimmed_right)
    )

    trimmed_left = trimmed_left[:min_len]
    trimmed_right = trimmed_right[:min_len]

    return np.stack(
        [trimmed_left, trimmed_right],
        axis=0
    )


def normalize_audio(audio):

    audio = audio.astype(np.float32)

    max_val = np.max(np.abs(audio))

    if max_val == 0:
        return audio

    return audio / max_val


def pad_or_crop_audio(audio, sample_rate, target_duration):

    target_length = int(target_duration * sample_rate)

    if audio.ndim == 1:
        current_length = len(audio)

        if current_length < target_length:
            amount_to_pad = target_length - current_length

            return np.pad(
                audio,
                (0, amount_to_pad),
                mode="constant"
            )

        return audio[:target_length]

    else:

        current_length = audio.shape[1]

        if current_length < target_length:

            amount_to_pad = target_length - current_length

            return np.pad(
                audio,
                ((0, 0), (0, amount_to_pad)),
                mode="constant"
            )

        return audio[:, :target_length]


# AUGMENTATION
def add_noise(audio, noise_level=0.003):

    noise = np.random.randn(*audio.shape)

    return (
        audio + noise_level * noise
    ).astype(np.float32)


def random_gain(audio, min_gain=0.8, max_gain=1.2):

    gain = random.uniform(min_gain, max_gain)

    return (audio * gain).astype(np.float32)


def random_shift(audio, max_shift=2000):

    shift = np.random.randint(-max_shift, max_shift)

    if audio.ndim == 1:
        return np.roll(audio, shift)

    return np.roll(audio, shift, axis=1)


def preprocess_audio(
    file_path,
    target_sr=16000,
    target_duration=None,
    trim=True,
    top_db=30,
    normalize=True,
    force_mono=False,
    augment=False
):

    audio, sample_rate = load_audio_file(
        file_path=file_path,
        target_sr=target_sr,
        mono=False
    )

    if force_mono:
        audio = convert_to_mono(audio)

    if trim:
        audio = trim_silence(audio, top_db=top_db)

    if normalize:
        audio = normalize_audio(audio)

    if augment:

        if random.random() < 0.5:
            audio = add_noise(audio)

        if random.random() < 0.5:
            audio = random_gain(audio)

        if random.random() < 0.5:
            audio = random_shift(audio)

    if target_duration is not None:
        audio = pad_or_crop_audio(
            audio,
            sample_rate,
            target_duration
        )

    return audio.astype(np.float32), sample_rate


def preprocess_dataset(
    dataset,
    target_sr=16000,
    target_duration=None,
    trim=True,
    top_db=30,
    normalize=True,
    force_mono=False,
    augment=True
):

    processed_dataset = []

    for entry in dataset:

        processed_audio, processed_sr = preprocess_audio(
            file_path=entry["file_path"],
            target_sr=target_sr,
            target_duration=target_duration,
            trim=trim,
            top_db=top_db,
            normalize=normalize,
            force_mono=force_mono,
            augment=augment
        )

        processed_entry = {
            "file_path": entry["file_path"],
            "original_label": entry.get("original_label"),
            "label": entry["label"],
            "audio": processed_audio,
            "sample_rate": processed_sr
        }

        processed_dataset.append(processed_entry)

    return processed_dataset