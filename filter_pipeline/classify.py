
import os

from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow_hub as hub

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
        scores = scores.numpy()
        
        speech_score = scores[:, self.speech_index].mean()
        music_score = scores[:, self.music_index].mean()
        
        return speech_score, music_score
    
    
