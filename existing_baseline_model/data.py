"""Manifest and batching helpers shared by this standalone baseline."""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class GlossManifest(Dataset):
    """JSONL rows: {"features": "clip.npy", "glosses": ["G1", "G2"]}."""
    def __init__(self, manifest, vocab):
        self.manifest = Path(manifest)
        self.root = self.manifest.parent
        self.rows = [json.loads(line) for line in self.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.vocab = vocab

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        x = np.load(self.root / row["features"]).astype(np.float32)
        if x.ndim != 2 or x.shape[1] != 276:
            raise ValueError(f"{row['features']} must have shape [T,276], got {x.shape}")
        y = [self.vocab[token] for token in row["glosses"]]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


def collate(batch):
    xs, ys = zip(*batch)
    lengths = torch.tensor([len(x) for x in xs], dtype=torch.long)
    targets = torch.cat(ys)
    xpad = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True)
    target_lengths = torch.tensor([len(y) for y in ys], dtype=torch.long)
    return xpad, lengths, targets, target_lengths


def load_vocab(path):
    vocab = json.loads(Path(path).read_text(encoding="utf-8"))
    if vocab.get("<blank>") != 0:
        raise ValueError("vocab.json must map <blank> to ID 0")
    return vocab
