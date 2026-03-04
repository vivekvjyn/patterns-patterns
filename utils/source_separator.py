from demucs.pretrained import get_model
from demucs.apply import apply_model
import numpy as np
import torch

class SourceSeparator:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.source_separator = get_model(name="htdemucs")
        self.source_separator.eval()
        self.source_separator.to(self.device)

    def __call__(self, audio, sample_rate):
        """Separates the vocal track from the given audio using the HT-Demucs algorithm.

        Args:
            audio (torch.Tensor or np.ndarray): The input audio signal. Can be a PyTorch tensor or a NumPy array.
            sample_rate (int): The sample rate of the input audio.
        Returns:
            vocals (np.ndarray): The separated vocal track as a NumPy array.
        """

        if type(audio) == np.ndarray:
            audio = torch.from_numpy(audio).float()

        audio = audio.unsqueeze(0).to(self.device)
        if audio.shape[1] == 1:
            audio = audio.repeat(1, 2, 1)

        with torch.no_grad():
            sources = apply_model(self.source_separator, audio, shifts=1, split=True)
            vocals = sources[0, 3].cpu().numpy()
            vocals = np.mean(vocals, axis=0)

        return vocals

    def __str__(self):
        return f"HT-Demucs | Device [{self.device}] | Callable: {self.__call__}"
