from compiam.melody.pitch_extraction import Melodia
import torch

class PitchExtractor:
    def __init__(self):
        self.pitch_extractor = Melodia()

    def __call__(self, audio, sample_rate):
        """Extracts the pitch track from the given audio using the Melodia algorithm.

        Args:
            audio (torch.Tensor or np.ndarray): The input audio signal. Can be a PyTorch tensor or a NumPy array.
            sample_rate (int): The sample rate of the input audio.
        Returns:
            times (np.ndarray): An array of time stamps corresponding to the pitch track.
            pitch (np.ndarray): An array of pitch values corresponding to the time stamps.
        """

        if type(audio) == torch.Tensor:
            audio = audio.cpu().numpy()

        pitch_track = self.pitch_extractor.extract(audio, sample_rate)
        times = pitch_track[:, 0]
        pitch = pitch_track[:, 1]
        return times, pitch

    def __str__(self):
        return f"Melodia | Callable: {self.__call__}"
