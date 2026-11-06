# Helper functions for x-vector shenanigans w/ speechbrain
from pathlib import Path
import torch
import torchaudio
from pympi.Elan import Eaf
from pandas import DataFrame

from speechbrain.inference.speaker import EncoderClassifier

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WINDOW_SIZE = 4.0 # these are in seconds btw
WINDOW_SLIDE = 2.0

# These are model-specific:

def get_classifier() -> EncoderClassifier:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-xvect-voxceleb",
        savedir="pretrained_models/spkrec-xvect-voxceleb",
        run_opts={"device": device}
    )
    print(f'using {device}')
    return classifier


def title_to_embeddings(title: str, classifier: EncoderClassifier, wav_dir: str = 'data/wav', eaf_dir: str = 'data/eaf') -> tuple[torch.Tensor, list[int]]:
    """
    Given the title of a file (e.g. the wav file w/o .wav), return a tuple in the form:
    (all of the embeddings from the file, a list of the times in milliseconds where those are from)
    """
    audio = load_file(Path(f'{wav_dir}/{title}.wav'))
    chunks = []
    times = []

    target_audio_snippets = get_snippets_from_eaf(Path(f'{eaf_dir}/{title}.eaf'))
    if len(target_audio_snippets) <= 0:
        # we just get out if an audio file doesn't have any snippets (usually when an interviewer is gabbing) 
        return (torch.tensor(-1), [-1])

    for interval in target_audio_snippets:
        snippet_audio, t = grab_snippet(audio, interval)
        new_chunk = chunk_audio_batch(snippet_audio)
        chunks.append(new_chunk)
        for i in range(new_chunk.size()[0]):
            times.append(int(t + (2000 * i) + (WINDOW_SIZE * (2000 / 2))))

    big_old_chunk = torch.cat(chunks, dim=0).squeeze(1)
    return (embed_batch(classifier, big_old_chunk), times)


def embed_file(classifier: EncoderClassifier, input: torch.Tensor):
    return classifier.encode_batch(input)[0][0]


def embed_batch(classifier: EncoderClassifier, input_batch: torch.Tensor):
    return classifier.encode_batch(input_batch).squeeze(1)


def avg_cos_similarity(encoded_block_1: torch.Tensor, encoded_block_2: torch.Tensor) -> float:
    """
    Real talk Google Gemini wrote this function because I knew it could be done faster than my for-loop nonsense
    """
    tensor_a_norm = torch.nn.functional.normalize(encoded_block_1, p=2, dim=1)
    tensor_b_norm = torch.nn.functional.normalize(encoded_block_2, p=2, dim=1)

    similarity_matrix = torch.matmul(tensor_a_norm, tensor_b_norm.T)

    return similarity_matrix.mean().item()


def save_embedded_batch(batch: tuple[torch.tensor, list[int]], title: str, output_dir: str = 'output') -> None:
    # before we do anything, we check if title_to_embeddings failed:
    if batch[1][0] == -1:
        print(f'file {title} had no target snippets :/')
        return

    # then we'll make sure that the folders exist:
    Path(f'{output_dir}/xvector').mkdir(parents = True, exist_ok = True)
    Path(f'{output_dir}/time').mkdir(parents = True, exist_ok = True)

    # store the xvectors (rows) as a csv!
    DataFrame(batch[0].numpy()).to_csv(f'{output_dir}/xvector/{title}', index = False)

    # hold on to the time info because probably it'll come up:
    DataFrame(batch[1]).to_csv(f'{output_dir}/time/{title}', index = False)
    

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


def chunk_audio_batch(audio: torch.Tensor, window_size_seconds: float = WINDOW_SIZE, window_slide_seconds: float = WINDOW_SLIDE, do_hamming: bool = False):
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

