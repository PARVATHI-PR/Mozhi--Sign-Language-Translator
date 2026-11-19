"""MSLR student with four K=5 1D convolutions, max pooling and BiLSTM."""
import torch
from torch import nn
import torch.nn.functional as F


class MSLRStudent(nn.Module):
    def __init__(self, vocab_size: int, input_dim: int = 276, channels: int = 256, hidden: int = 256):
        super().__init__()
        widths = [input_dim, channels, channels, channels, channels]
        layers = []
        for a, b in zip(widths[:-1], widths[1:]):
            layers.extend([nn.Conv1d(a, b, kernel_size=5, padding=2), nn.BatchNorm1d(b), nn.ReLU()])
        self.convs = nn.Sequential(*layers)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, ceil_mode=True)
        self.bilstm = nn.LSTM(channels, hidden, num_layers=2, batch_first=True,
                              bidirectional=True, dropout=0.25)
        self.conv_projection = nn.Linear(channels, hidden * 2)
        self.classifier = nn.Linear(hidden * 2, vocab_size)

    def forward(self, x, lengths=None):
        # Input [B,T,276]; probabilities are [B,T',V] at both distillation levels.
        conv_features = self.convs(x.transpose(1, 2))
        conv_features = self.pool(conv_features).transpose(1, 2)
        conv_probs = self.classifier(self.conv_projection(conv_features)).softmax(-1)
        recurrent, _ = self.bilstm(conv_features)
        lstm_logits = self.classifier(recurrent)
        lstm_probs = lstm_logits.softmax(-1)
        out_lengths = None if lengths is None else torch.div(lengths + 1, 2, rounding_mode="floor")
        return {"conv_probs": conv_probs, "lstm_probs": lstm_probs,
                "lstm_logits": lstm_logits, "output_lengths": out_lengths}
