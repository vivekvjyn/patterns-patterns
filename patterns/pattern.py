import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from utils.dataset import Dataset
from sklearn.metrics.pairwise import cosine_similarity

class Pattern:
    def __init__(self, source_separator, pitch_extractor, tonic_identifier, embedder):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.source_separator = source_separator
        self.pitch_extractor = pitch_extractor
        self.tonic_identifier = tonic_identifier
        self.embedder = embedder

    def __call__(self, audio_path):
        audio, sample_rate = torchaudio.load(audio_path)

        vocals = self.source_separator(audio, sample_rate)

        times, pitch = self.pitch_extractor(vocals, sample_rate)

        tonic = self.tonic_identifier(audio, sample_rate)

        pitch_cents = self.hz_to_cents(pitch, tonic)

        splits, num_splits = self.split_pitch(pitch_cents)

        dataset = Dataset(splits)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=num_splits, shuffle=False)
        embeddings = self.embedder(data_loader)

        self_similarity_matrix = self.compute_self_similarity(embeddings)

        self.visualize_self_similarity(self_similarity_matrix, pitch_cents, times)


    def hz_to_cents(self, frequencies, tonic):
        pitch_cents = 1200 * np.log2(frequencies / tonic)
        pitch_cents[~np.isfinite(pitch_cents)] = np.nan

        return pitch_cents

    def split_pitch(self, x, window_size=100):
        num_splits = len(x) // window_size
        split_size = len(x) // num_splits
        splits = [x[i*split_size : (i+1)*split_size] for i in range(num_splits)]

        normalized_splits = self.normalize(splits)

        return normalized_splits, num_splits

    def normalize(self, data, range_min=-4200, range_max=4200):
        normalized_data = []
        for sample in data:
            normalized_sample = (sample - range_min) / (range_max - range_min)
            normalized_data.append(normalized_sample)
        return np.array(normalized_data)

    def compute_self_similarity(self, embeddings):
        similarity_matrix = cosine_similarity(embeddings)
        return similarity_matrix

    def visualize_self_similarity(self, similarity_matrix, pitch, times):
        plt.figure(figsize=(18, 18))

        # First subplot: Flipped pitch vs time
        plt.subplot(2, 2, 1)
        plt.plot(np.flip(pitch), np.flip(times))
        plt.xticks([])
        plt.yticks([])

        # Second subplot: Self-similarity matrix
        plt.subplot(2, 2, 2)
        plt.imshow(np.flip(similarity_matrix, 0), cmap='viridis', aspect='auto',
                   extent=[times[0], times[-1], times[0], times[-1]])
        plt.title('Self-Similarity Matrix')
        plt.xlabel('Time (s)')
        plt.ylabel('Time (s)')

        # Third subplot: (Empty or other content, if needed)
        plt.subplot(2, 2, 3)
        plt.axis('off')  # Hide if not used

        # Fourth subplot: Flipped pitch vs time
        plt.subplot(2, 2, 4)
        plt.plot(times, pitch)
        plt.xticks([])
        plt.yticks([])

        plt.tight_layout()
        plt.savefig("self_similarity_matrix.png")
        plt.close()
