from compiam.melody.tonic_identification import TonicIndianMultiPitch
import numpy as np
import torch

class TonicIdentifier:
    def __init__(self):
        self.tonic_identifier = TonicIndianMultiPitch()

    def __call__(self, audio, sample_rate):
        """Identifies the tonic of the given audio using the TonicIndianMultiPitch algorithm.

        Args:
            audio (torch.Tensor or np.ndarray): The input audio signal. Can be a PyTorch tensor or a NumPy array.
            sample_rate (int): The sample rate of the input audio.
        Returns:
            tonic (float): The identified tonic frequency in Hz.
        """

        if type(audio) == torch.Tensor:
            audio = audio.cpu().numpy()

        tonic = self.tonic_identifier.extract(audio, sample_rate)
        return tonic

    def __str__(self):
        return f"TonicIndianMultiPitch | Callable: {self.__call__}"
