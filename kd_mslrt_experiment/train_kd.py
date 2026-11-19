"""Train MSLR using frozen video teacher and multi-level KD + CTC."""
import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "existing_baseline_model"))
from data import GlossManifest, collate, load_vocab
from models.augmentation import augment_landmarks
from models.student import MSLRStudent
from models.teacher import CorrNetStyleTeacher
from kd_loss import kd_mslrt_loss


class VideoGlossManifest(GlossManifest):
    """Rows additionally contain video: .npy float [T,3,H,W], RGB normalized [0,1]."""
    def __init__(self, manifest, vocab, augment=False):
        super().__init__(manifest, vocab)
        self.augment = augment

    def __getitem__(self, i):
        x, y = super().__getitem__(i)
        row = self.rows[i]
        import numpy as np
        video = torch.from_numpy(np.load(self.root / row["video"]).astype("float32") / 255.0)
        if video.ndim != 4 or video.shape[1] != 3:
            raise ValueError(f"{row['video']} must have shape [T,3,H,W]")
        video = torch.nn.functional.interpolate(video[::2], size=(64, 64), mode="bilinear", align_corners=False)
        return (augment_landmarks(x) if self.augment else x), video, y


def collate_kd(batch):
    xs, videos, ys = zip(*batch)
    lengths = torch.tensor([len(x) for x in xs]); target_lengths = torch.tensor([len(y) for y in ys])
    xpad = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True)
    vpad = torch.nn.utils.rnn.pad_sequence(videos, batch_first=True)
    return xpad, vpad, lengths, torch.cat(ys), target_lengths


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True); p.add_argument("--dev", required=True); p.add_argument("--vocab", required=True)
    p.add_argument("--teacher", required=True, help="Pretrained compatible CorrNet-style teacher checkpoint")
    p.add_argument("--epochs", type=int, default=80); p.add_argument("--output", default="checkpoints/best.pt")
    p.add_argument("--temperature", type=float, default=1.0); p.add_argument("--kd-weight", type=float, default=25.0)
    a = p.parse_args(); vocab = load_vocab(a.vocab); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher = CorrNetStyleTeacher(len(vocab)).to(device)
    teacher.load_state_dict(torch.load(a.teacher, map_location=device, weights_only=True))
    teacher.eval().requires_grad_(False)
    student = MSLRStudent(len(vocab)).to(device)
    train = DataLoader(VideoGlossManifest(a.train, vocab, augment=True), batch_size=4, shuffle=True, collate_fn=collate_kd)
    dev = DataLoader(VideoGlossManifest(a.dev, vocab), batch_size=4, collate_fn=collate_kd)
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-4, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1., end_factor=.01, total_iters=a.epochs)
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True); best = float("inf"); history = []
    for epoch in range(1, a.epochs + 1):
        row = {}
        for training, loader in ((True, train), (False, dev)):
            student.train(training); totals = {k: 0. for k in ("total", "ctc", "kd", "lc", "lb", "ls")}; n = 0
            for x, video, lengths, targets, target_lengths in loader:
                x, video, targets = x.to(device), video.to(device), targets.to(device)
                with torch.set_grad_enabled(training):
                    with torch.no_grad(): teacher_out = teacher(video)
                    student_out = student(x, lengths)
                    losses = kd_mslrt_loss(student_out, teacher_out, targets, lengths, target_lengths,
                                           temperature=a.temperature, kd_weight=a.kd_weight)
                    if training:
                        optimizer.zero_grad(set_to_none=True); losses["total"].backward()
                        torch.nn.utils.clip_grad_norm_(student.parameters(), 5.); optimizer.step()
                for k in totals: totals[k] += float(losses[k].detach())
                n += 1
            for k in totals: row[f"{'train' if training else 'dev'}_{k}"] = totals[k] / max(n, 1)
        schedule.step(); history.append(row); print(f"epoch={epoch:03d} {row}")
        if row["dev_total"] < best:
            best = row["dev_total"]; torch.save(student.state_dict(), out)
    (out.parent / "training_metrics.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__": main()
