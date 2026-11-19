"""Plain landmark CTC baseline: 1D CNN -> BiGRU -> gloss logits."""
import torch
from torch import nn


class LandmarkCTCBaseline(nn.Module):
    def __init__(self, vocab_size: int, input_dim: int = 276, hidden: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, 256, 5, padding=2), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Conv1d(256, 256, 5, padding=2), nn.BatchNorm1d(256), nn.ReLU(),
        )
        self.temporal = nn.GRU(256, hidden, num_layers=2, batch_first=True,
                               bidirectional=True, dropout=0.25)
        self.classifier = nn.Linear(2 * hidden, vocab_size)

    def forward(self, x, lengths=None):
        x = self.encoder(x.transpose(1, 2)).transpose(1, 2)
        x, _ = self.temporal(x)
        return self.classifier(x)
