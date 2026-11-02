# Helper functions for x-vector shenanigans w/ speechbrain
from pathlib import Path
import torch
import torchaudio

from speechbrain.inference.speaker import EncoderClassifier

# These are model-specific:

def classifier() -> EncoderClassifier:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-xvect-voxceleb",
        savedir="pretrained_models/spkrec-xvect-voxceleb",
        run_opts={"device": device}
    )
    print(f'using {device}')
    return classifier


def embed_file(classifier, input):
    return classifier.encode_batch(input)[0][0]




# These are general helpers:

def load_file(path):
    wav, sr = torchaudio.load(path)
    return torchaudio.transforms.Resample(sr, 16000)(wav)

