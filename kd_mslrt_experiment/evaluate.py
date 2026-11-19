"""Evaluate the distilled MSLR student with the same greedy CTC WER as baseline."""
import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "existing_baseline_model"))
from data import GlossManifest, collate, load_vocab
from models.student import MSLRStudent
from existing_baseline_model.evaluate import evaluate


def main():
    p = argparse.ArgumentParser(); p.add_argument("--manifest", required=True); p.add_argument("--vocab", required=True)
    p.add_argument("--checkpoint", default="checkpoints/best.pt"); p.add_argument("--output", default="metrics.json")
    a = p.parse_args(); vocab = load_vocab(a.vocab); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MSLRStudent(len(vocab)).to(device); model.load_state_dict(torch.load(a.checkpoint, map_location=device, weights_only=True))
    # Adapt the student head to the shared evaluator interface (greedy CTC decode).
    class CTCAdapter(torch.nn.Module):
        def __init__(self, wrapped): super().__init__(); self.wrapped = wrapped
        def forward(self, x): return self.wrapped(x)["lstm_logits"]
        @staticmethod
        def transform_lengths(lengths): return torch.div(lengths + 1, 2, rounding_mode="floor")
    metrics = evaluate(CTCAdapter(model), DataLoader(GlossManifest(a.manifest, vocab), batch_size=8,
                     collate_fn=collate), {v:k for k,v in vocab.items()}, device)
    metrics["accuracy"] = 1.0 - metrics["wer"]
    Path(a.output).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__": main()
