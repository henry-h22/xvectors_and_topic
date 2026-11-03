# Helper functions for x-vector shenanigans w/ speechbrain
from pathlib import Path
import torch
import torchaudio
from pympi.Elan import Eaf

from speechbrain.inference.speaker import EncoderClassifier

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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


def embed_file(classifier: EncoderClassifier, input: torch.Tensor):
    return classifier.encode_batch(input)[0][0]


def embed_batch(classifier: EncoderClassifier, input_batch: torch.Tensor):
    return classifier.encode_batch(input_batch)


# these are audio helpers:

def load_file(path: Path):
    wav, sr = torchaudio.load(path)
    return torchaudio.transforms.Resample(sr, 16000)(wav)


def ms_to_samples(ms: int, sr: int = 16000) -> int:
    return int((ms * sr) / 1000)


def grab_snippet(audio: torch.tensor, boundaries_ms: tuple[int, int]) -> tuple[torch.tensor, int]:
    """
    Given a torchaudio tensor @ sample rate 16,000 Hz, returns a tuple:
    (just the bit of audio between the two bounds,
    the start of the snippet in ms w.r.t. the entire audio file)
    """

    start_sample = ms_to_samples(boundaries_ms[0])
    if start_sample < 0: start_sample = 0

    end_sample = ms_to_samples(boundaries_ms[1])
    max_sample = audio.size()[1] - 1
    if end_sample > max_sample: end_sample = max_sample

    return (audio[0][start_sample:end_sample], boundaries_ms[0])


def chunk_audio_batch(audio: torch.Tensor, window_size_seconds: float = 4.0, window_slide_seconds: float = 2.0, do_hamming: bool = True):
    window_size_samples = round(window_size_seconds * 16000)
    window_slide_samples = round(window_slide_seconds * 16000)

    chunks = audio.unfold(0, window_size_samples, window_slide_samples)

    if do_hamming:
        hamming = torch.hamming_window(
            window_size_samples,
            periodic = False,
            device = DEVICE
        )
        chunks = chunks * hamming

    return chunks


# These are eaf-side helpers:

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

