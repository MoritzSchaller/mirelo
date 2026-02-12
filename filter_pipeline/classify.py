
import os

from typing import Iterable
from pathlib import Path
from multiprocessing import Pool, cpu_count

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow_hub as hub
import pandas as pd
import numpy as np

from tqdm import tqdm

from filter_pipeline.audio_io import load_audio_av


class AudioClassifier:
    def __init__(self):
        self.model = hub.load('https://tfhub.dev/google/yamnet/1')
        self.sr = 16000
        self.music_index = 132
        self.speech_index = 0

    def process_file(self, path: Path) -> dict[str]:
        """Run music and speech classification on a single audio file."""
        
        audio = load_audio_av(path, self.sr)
        
        scores, embeddings, spectrogram = self.model(audio)
        scores: np.ndarray = scores.numpy()
        
        speech_score = scores[:, self.speech_index] # ~20 scores
        music_score = scores[:, self.music_index]
        
        speech_score = _mean_of_n_highest(speech_score, 3)
        music_score = _mean_of_n_highest(music_score, 3)

        return speech_score, music_score
    

def classify_multiprocessed(sources: Iterable[Path|str|bytes]) -> pd.DataFrame:
    with Pool(cpu_count(), _init_worker) as p:
        scores = tuple(tqdm(
            p.imap(_process_wrapper, sources),
            desc="Classifying Audio Files",
            total=len(sources)
        ))

    df = pd.DataFrame.from_records(scores, columns=("speech_score", "music_score"))
    return df


def _init_worker():
    global classifier 
    classifier = AudioClassifier()


def _process_wrapper(path):
    return classifier.process_file(path)


def _mean_of_n_highest(x: np.ndarray, n: int=5) -> float:
    return float(np.sort(x)[-5:].mean())