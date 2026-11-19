"""Video CorrNet-style teacher interface.

This is a trainable 3D-CNN + temporal BiLSTM teacher scaffold. The paper uses
CorrNet; exact CorrNet weights/implementation must be supplied by the user.
The KD script requires a compatible pretrained checkpoint before student training.
"""
import torch
from torch import nn


class CorrNetStyleTeacher(nn.Module):
    def __init__(self, vocab_size: int, width: int = 64, hidden: int = 256):
        super().__init__()
        self.video_encoder = nn.Sequential(
            nn.Conv3d(3, width, (3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False),
            nn.BatchNorm3d(width), nn.ReLU(inplace=True), nn.MaxPool3d((1, 3, 3), (1, 2, 2), (0, 1, 1)),
            nn.Conv3d(width, width * 2, 3, stride=(1, 2, 2), padding=1, bias=False),
            nn.BatchNorm3d(width * 2), nn.ReLU(inplace=True),
            nn.Conv3d(width * 2, width * 4, 3, stride=(1, 2, 2), padding=1, bias=False),
            nn.BatchNorm3d(width * 4), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((None, 1, 1)),
        )
        self.temporal = nn.LSTM(width * 4, hidden, num_layers=2, batch_first=True,
                                bidirectional=True, dropout=0.25)
        self.conv_projection = nn.Linear(width * 4, hidden * 2)
        self.classifier = nn.Linear(hidden * 2, vocab_size)

    def forward(self, video):
        # video [B,T,3,H,W]. CorrNet/MSLR use a shared gloss classifier.
        z = self.video_encoder(video.transpose(1, 2)).squeeze(-1).squeeze(-1).transpose(1, 2)
        conv_logits = self.classifier(self.conv_projection(z))
        conv_probs = conv_logits.softmax(-1)
        z, _ = self.temporal(z)
        lstm_logits = self.classifier(z)
        lstm_probs = lstm_logits.softmax(-1)
        return {"conv_probs": conv_probs, "lstm_probs": lstm_probs,
                "lstm_logits": lstm_logits}
