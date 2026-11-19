"""Diagnostic greedy CTC WER for the locally trained video teacher."""
import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "existing_baseline_model"))
from data import load_vocab
from existing_baseline_model.evaluate import evaluate
from models.teacher import CorrNetStyleTeacher
from train_teacher import VideoDataset, collate


class CTCAdapter(torch.nn.Module):
    def __init__(self, wrapped): super().__init__(); self.wrapped = wrapped
    def forward(self, x): return self.wrapped(x)["lstm_logits"]


def main():
    p = argparse.ArgumentParser(); p.add_argument("--manifest", required=True); p.add_argument("--vocab", required=True)
    p.add_argument("--checkpoint", default="checkpoints/teacher.pt"); p.add_argument("--output", default="teacher_metrics.json")
    a = p.parse_args(); vocab = load_vocab(a.vocab); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher = CorrNetStyleTeacher(len(vocab)).to(device)
    teacher.load_state_dict(torch.load(a.checkpoint, map_location=device, weights_only=True))
    loader = DataLoader(VideoDataset(a.manifest, vocab), batch_size=2, collate_fn=collate)
    metrics = evaluate(CTCAdapter(teacher), loader, {v:k for k,v in vocab.items()}, device)
    Path(a.output).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__": main()
