import torch
from demucs.pretrained import get_model
from demucs.apply import apply_model
import torchaudio

class Pattern:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.separator = get_model(name="htdemucs")
        self.separator.eval()
        self.separator.to(self.device)

    def __call__(self, audio_path):
        audio, sample_rate = torchaudio.load(audio_path)

        sources = self._separate(audio, sample_rate)

        return sources

    def _separate(self, audio, sample_rate):
        audio = audio.unsqueeze(0).to(self.device)

        if audio.shape[1] == 1:
            audio = audio.repeat(1, 2, 1)

        with torch.no_grad():
            sources = apply_model(self.separator, audio, shifts=1, split=True)

        sources = sources[0]

        for i, name in enumerate(self.separator.sources):
            torchaudio.save(f"{name}.wav", sources[i].cpu(), sample_rate)

        return sources
