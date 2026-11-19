"""Extract keypoints from the ISL-CSLRT sentence videos and train a GRU classifier."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from isl_model import FEATURES, SEQ_LEN, make_landmarkers, sentence_folders, video_sequence


class SentenceGRU(nn.Module):
    def __init__(self, classes: int):
        super().__init__()
        self.input_norm = nn.LayerNorm(FEATURES)
        self.gru = nn.GRU(FEATURES, 128, num_layers=2, batch_first=True,
                          bidirectional=True, dropout=0.25)
        self.head = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, classes))

    def forward(self, x):
        x = self.input_norm(x)
        y, _ = self.gru(x)
        return self.head(y.mean(dim=1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Kaggle dataset version folder")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--limit-per-class", type=int, default=0, help="0 means use all clips")
    parser.add_argument("--output", default="artifacts")
    args = parser.parse_args()
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    root = sentence_folders(Path(args.dataset))
    labels = sorted(p.name for p in root.iterdir() if p.is_dir())
    label_to_id = {label: i for i, label in enumerate(labels)}
    clips = []
    for label in labels:
        paths = sorted(p for p in (root / label).rglob("*") if p.suffix.lower() in {".mp4", ".mov", ".avi"})
        if args.limit_per_class:
            paths = paths[:args.limit_per_class]
        clips.extend((p, label_to_id[label]) for p in paths)
    if len(clips) < 2:
        raise RuntimeError(f"Only found {len(clips)} videos in {root}")
    print(f"Found {len(clips)} videos across {len(labels)} sentence labels.")
    hand, pose = make_landmarkers(Path("models"))
    X, y = [], []
    try:
        for i, (path, label_id) in enumerate(clips, 1):
            try:
                X.append(video_sequence(path, hand, pose))
                y.append(label_id)
            except Exception as exc:
                print(f"Skipping {path.name}: {exc}")
            if i % 25 == 0 or i == len(clips):
                print(f"Extracted {i}/{len(clips)}")
    finally:
        hand.close()
        pose.close()
    if len(X) < 2:
        raise RuntimeError("Could not extract enough videos")
    X = torch.tensor(np.stack(X), dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    # Stratified-by-class holdout where possible; clips are grouped by phrase, not signer.
    indices = np.arange(len(y))
    rng = np.random.default_rng(42)
    train_idx, val_idx = [], []
    for class_id in range(len(labels)):
        group = indices[y.numpy() == class_id]
        rng.shuffle(group)
        n_val = max(1, round(len(group) * 0.2)) if len(group) > 1 else 0
        val_idx.extend(group[:n_val])
        train_idx.extend(group[n_val:])
    if not val_idx:
        val_idx = train_idx[-1:]
    train_dl = DataLoader(TensorDataset(X[train_idx], y[train_idx]), batch_size=16, shuffle=True)
    val_dl = DataLoader(TensorDataset(X[val_idx], y[val_idx]), batch_size=32)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceGRU(len(labels)).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    best = 0.0
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        for bx, by in train_dl:
            bx, by = bx.to(device), by.to(device)
            optim.zero_grad(set_to_none=True)
            loss_fn(model(bx), by).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for bx, by in val_dl:
                pred = model(bx.to(device)).argmax(1).cpu()
                correct += int((pred == by).sum())
                total += len(by)
        acc = correct / max(total, 1)
        print(f"Epoch {epoch:02d}/{args.epochs}  validation clip accuracy {acc:.1%}")
        if acc >= best:
            best = acc
            torch.save({"state_dict": model.cpu().state_dict(), "classes": labels,
                        "seq_len": SEQ_LEN, "features": FEATURES}, out / "isl_gru.pt")
            model.to(device)
    (out / "labels.json").write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {out / 'isl_gru.pt'}; best validation clip accuracy {best:.1%}.")
    print("Validation split is by clip. It does not establish unseen-signer generalization.")


if __name__ == "__main__":
    main()
