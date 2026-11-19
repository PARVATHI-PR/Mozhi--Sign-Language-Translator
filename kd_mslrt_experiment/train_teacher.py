"""Train the video teacher scaffold with gloss CTC before student distillation."""
import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "existing_baseline_model"))
from data import GlossManifest, load_vocab
from models.teacher import CorrNetStyleTeacher


class VideoDataset(GlossManifest):
    def __getitem__(self, i):
        import numpy as np
        row = self.rows[i]
        x = torch.from_numpy(np.load(self.root / row["video"]).astype(np.float32) / 255.0)
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f"{row['video']} must have shape [T,3,H,W]")
        x = torch.nn.functional.interpolate(x[::2], size=(64, 64), mode="bilinear", align_corners=False)
        y = torch.tensor([self.vocab[t] for t in row["glosses"]], dtype=torch.long)
        return x, y


def collate(batch):
    xs, ys = zip(*batch); lengths = torch.tensor([len(x) for x in xs])
    target_lengths = torch.tensor([len(y) for y in ys])
    return torch.nn.utils.rnn.pad_sequence(xs, batch_first=True), lengths, torch.cat(ys), target_lengths


def epoch(model, loader, optimizer, device):
    training = optimizer is not None; model.train(training); ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    total = count = 0
    for video, lengths, targets, target_lengths in loader:
        video, targets = video.to(device), targets.to(device)
        with torch.set_grad_enabled(training):
            logits = model(video)["lstm_logits"]
            loss = ctc(logits.log_softmax(-1).transpose(0, 1), targets, lengths, target_lengths)
            if training:
                optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.); optimizer.step()
        total += float(loss.detach()); count += 1
    return total / max(1, count)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--train", required=True); p.add_argument("--dev", required=True)
    p.add_argument("--vocab", required=True); p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=2); p.add_argument("--output", default="checkpoints/teacher.pt")
    a = p.parse_args(); vocab = load_vocab(a.vocab); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = DataLoader(VideoDataset(a.train, vocab), batch_size=a.batch_size, shuffle=True, collate_fn=collate)
    dev = DataLoader(VideoDataset(a.dev, vocab), batch_size=a.batch_size, collate_fn=collate)
    model = CorrNetStyleTeacher(len(vocab)).to(device); opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1., end_factor=.01, total_iters=a.epochs)
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True); best = float("inf"); history = []
    for e in range(1, a.epochs + 1):
        tr, dv = epoch(model, train, opt, device), epoch(model, dev, None, device); sched.step()
        history.append({"epoch": e, "train_ctc": tr, "dev_ctc": dv}); print(f"epoch={e:03d} train_ctc={tr:.4f} dev_ctc={dv:.4f}")
        if dv < best:
            best = dv; torch.save(model.state_dict(), out)
    (out.parent / "teacher_training_metrics.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__": main()
