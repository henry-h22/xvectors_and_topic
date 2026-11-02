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

def load_file(path: Path):
    wav, sr = torchaudio.load(path)
    return torchaudio.transforms.Resample(sr, 16000)(wav)


def ms_to_samples(ms: int, sr: int = 16000):
    return (ms * sr) / 1000


def grab_snippet(audio: torch.tensor, start_ms: int, end_ms: int) -> torch.tensor:
    """
    Given a torchaudio tensor @ sample rate 16,000 Hz, return just the bit between the two bounds
    """

    start_sample = ms_to_samples(start_ms)
    if start_sample < 0: start_sample = 0

    end_sample = ms_to_samples(end_ms)
    max_sample = audio.size()[1] - 1
    if end_sample > max_sample: end_sample = max_sample

    return audio[ms_to_samples(start_ms):]

