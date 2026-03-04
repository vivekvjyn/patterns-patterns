import numpy as np
import torch
from model.model import Model
import os

class Embedder:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedder = Model(embed_dim=48, depth=5).to(self.device)
        self.embedder.encoder.load(os.path.join("checkpoints", "encoder.pth"), self.device)

    def __call__(self, data_loader):
        embeddings = self._propagate(data_loader)
        return embeddings

    def _propagate(self, data_loader):
        self.embedder.eval()

        embeddings = np.array([])
        for i, (input) in enumerate(data_loader):
            embedding = self._embed(input).detach().cpu().numpy()
            embeddings = np.concatenate((embeddings, embedding), axis=0) if embeddings.size else embedding

        return embeddings

    def _embed(self, pitch):
        silence_mask = (torch.isnan(pitch)).float()
        pitch = torch.nan_to_num(pitch, nan=0)
        input = torch.cat([pitch, silence_mask], dim=1)

        return self.embedder(input)
