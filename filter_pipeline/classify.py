import numpy as np
import torch

from transformers import ClapModel, ClapProcessor


class HtsatAudioClassifier:
    def __init__(self, labels: list[str] = ("music", "speech")):
        self.labels = labels
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sr = 48000
        self.processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
        self.model = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(self.device)
        self.model.eval()

        # Precompute text embeddings
        with torch.no_grad():
            text_inputs = self.processor(
                text=self.labels,
                return_tensors="pt",
                padding=True,
            ).to(self.device)
            self.text_embeds = self.model.get_text_features(**text_inputs).pooler_output

    @torch.no_grad()
    def classify_batch(self, audios: list[np.ndarray]):
        audio_inputs = self.processor(
            audio=audios,
            sampling_rate=self.sr,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        audio_embeds = self.model.get_audio_features(**audio_inputs).pooler_output 
        logit_scale_audio = self.model.logit_scale_a.exp()
        logits_per_audio = torch.matmul(audio_embeds, self.text_embeds.t()) * logit_scale_audio

        return [
            {label: float(logit) for label, logit in zip(self.labels, row)}
            for row in logits_per_audio
        ]
    