import numpy as np

from preprocessing import preprocess_audio
from feature_extract import extract_features_from_audio
from direction_model import load_model_bundle, predict_one


def predictWavFile(wavPath):
    audio, sampleRate = preprocess_audio(
        file_path=wavPath,
        target_sr=16000,
        target_duration=2.0,
        silence_threshold=500
    )

    featureVector = extract_features_from_audio(audio, sampleRate)
    featureVector = np.asarray(featureVector, dtype=np.float32)

    bundle = load_model_bundle()
    model = bundle["model"]
    labelEncoder = bundle["label_encoder"]

    return predict_one(model, featureVector, labelEncoder)