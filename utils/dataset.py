import torch

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sequences = torch.tensor(data, dtype=torch.float32).to(self.device)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx].unsqueeze(0)
        return sequence

    def __str__(self):
        num_samples = len(self)
        return f"num_samples={num_samples}"
