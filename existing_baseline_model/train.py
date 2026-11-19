"""Train the no-teacher, no-distillation 1D-CNN -> BiGRU -> CTC baseline."""
import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from data import GlossManifest, collate, load_vocab
from model import LandmarkCTCBaseline


def run_epoch(model, loader, optimizer, device):
    training = optimizer is not None; model.train(training)
    loss_fn = nn.CTCLoss(blank=0, zero_infinity=True); total = count = 0
    for x, lengths, targets, target_lengths in loader:
        x, targets = x.to(device), targets.to(device)
        logits = model(x); logp = logits.log_softmax(-1).transpose(0, 1)
        loss = loss_fn(logp, targets, lengths, target_lengths)
        if training:
            optimizer.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
        total += float(loss.detach()); count += 1
    return total / max(count, 1)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--train", required=True); p.add_argument("--dev", required=True)
    p.add_argument("--vocab", required=True); p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--output", default="checkpoints/best.pt"); a = p.parse_args()
    vocab = load_vocab(a.vocab); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = DataLoader(GlossManifest(a.train, vocab), batch_size=8, shuffle=True, collate_fn=collate)
    dev = DataLoader(GlossManifest(a.dev, vocab), batch_size=8, collate_fn=collate)
    model = LandmarkCTCBaseline(len(vocab)).to(device); opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1.0, end_factor=0.01, total_iters=a.epochs)
    best = float("inf"); dest = Path(a.output); dest.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, a.epochs + 1):
        tr = run_epoch(model, train, opt, device); dv = run_epoch(model, dev, None, device); schedule.step()
        print(f"epoch={epoch:03d} train_ctc={tr:.4f} dev_ctc={dv:.4f}")
        if dv < best:
            best = dv; torch.save(model.state_dict(), dest)
    (dest.parent / "training_metrics.json").write_text(json.dumps({"best_dev_ctc": best}), encoding="utf-8")


if __name__ == "__main__": main()
