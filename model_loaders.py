# Helper functions for x-vector shenanigans w/ speechbrain
from pathlib import Path
import torch
import torchaudio
from pympi.Elan import Eaf

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


def eaf_pair_to_snippet_times(eaf_code_1: tuple[int, int, str], eaf_code_2: tuple[int, int, str]) -> tuple[int, int]:
    return (int((eaf_code_1[0] + eaf_code_1[1]) / 2), int((eaf_code_2[0] + eaf_code_2[1]) / 2))


def get_snippets_from_eaf(path: Path, p2_threshold: int = 3):
    eaf_file = Eaf(file_path = path)
    tier_names = list(eaf_file.get_tier_names())
    gaps_generator = eaf_file.get_gaps_and_overlaps2(tier_names[0], tier_names[1], maxlen=-1)

    answers = []
    last = next(gaps_generator)

    while True:
        try:
            current = next(gaps_generator)
        except StopIteration:
            break

        if current[2] == 'P2':
            start = last

            p2s = 0
            while current[2] == 'P2':
                try:
                    current = next(gaps_generator)
                    p2s += 1
                except StopIteration:
                    # note that in cases where the file "ends" w/ a P2, we just lose a little bit off the end because we call the last P2 the "end" code
                    break

            end = current
            if p2s > p2_threshold: answers.append(eaf_pair_to_snippet_times(start, end))
            # todo: one day we'll like combine the smaller ones into bigger spans i assume? for the topic models or wtv

        last = current

    return answers

