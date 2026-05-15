import numpy as np
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROCESSOR = Wav2Vec2Processor.from_pretrained(
    "facebook/wav2vec2-base-960h"
)

WAV2VEC_MODEL = Wav2Vec2Model.from_pretrained(
    "facebook/wav2vec2-base-960h"
)

WAV2VEC_MODEL.to(DEVICE)
WAV2VEC_MODEL.eval()


def extract_channel_features(channel_audio, sample_rate):

    inputs = PROCESSOR(
        channel_audio,
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding=True
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = WAV2VEC_MODEL(**inputs)

    hidden_states = outputs.last_hidden_state

    mean_pool = hidden_states.mean(dim=1)
    max_pool, _ = hidden_states.max(dim=1)
    std_pool = hidden_states.std(dim=1)

    pooled = torch.cat(
        [mean_pool, max_pool, std_pool],
        dim=1
    )

    return pooled.squeeze(0)


def extract_wav2vec_features(audio, sample_rate):

    if sample_rate != 16000:
        raise ValueError(
            f"Wav2Vec2 expects 16000 Hz audio, got {sample_rate}"
        )

    audio = np.asarray(audio, dtype=np.float32)

    # Stereo
    if audio.ndim == 2:

        left_channel = audio[0]
        right_channel = audio[1]

        left_features = extract_channel_features(
            left_channel,
            sample_rate
        )

        right_features = extract_channel_features(
            right_channel,
            sample_rate
        )

        feature_vector = torch.cat(
            [left_features, right_features],
            dim=0
        )

    # Mono
    else:

        audio = audio.flatten()

        feature_vector = extract_channel_features(
            audio,
            sample_rate
        )

    return feature_vector.cpu().numpy().astype(np.float32)