"""Evaluate a trained baseline checkpoint on a gloss CTC manifest."""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data import GlossManifest, collate, load_vocab
from model import LandmarkCTCBaseline


def edit_counts(ref, hyp):
    # Dynamic programming over (total edits, substitutions, insertions, deletions).
    prev = [(j, 0, j, 0) for j in range(len(hyp) + 1)]
    for i, ref_token in enumerate(ref, 1):
        row = [(i, 0, 0, i)]
        for j, hyp_token in enumerate(hyp, 1):
            if ref_token == hyp_token:
                row.append(prev[j - 1])
                continue
            candidates = [
                (prev[j - 1][0] + 1, prev[j - 1][1] + 1, prev[j - 1][2], prev[j - 1][3]),
                (row[j - 1][0] + 1, row[j - 1][1], row[j - 1][2] + 1, row[j - 1][3]),
                (prev[j][0] + 1, prev[j][1], prev[j][2], prev[j][3] + 1),
            ]
            row.append(min(candidates, key=lambda value: value[0]))
        prev = row
    _, substitutions, insertions, deletions = prev[-1]
    return substitutions, insertions, deletions


def decode(logits, lengths, id_to_token):
    out = []
    for row, n in zip(logits.argmax(-1), lengths):
        seq, prev = [], None
        for token in row[:int(n)].tolist():
            if token and token != prev:
                seq.append(id_to_token[token])
            prev = token
        out.append(seq)
    return out


def evaluate(model, loader, id_to_token, device):
    model.eval(); S = I = D = N = exact = samples = 0
    with torch.no_grad():
        for x, lengths, targets, target_lengths in loader:
            x = x.to(device)
            logits = model(x)
            if hasattr(model, "transform_lengths"):
                lengths = model.transform_lengths(lengths)
            predictions = decode(logits.cpu(), lengths, id_to_token)
            offset = 0
            for pred, n in zip(predictions, target_lengths.tolist()):
                ref = [id_to_token[i] for i in targets[offset:offset+n].tolist()]
                offset += n
                s, i, d = edit_counts(ref, pred); S += s; I += i; D += d; N += len(ref)
                exact += int(ref == pred); samples += 1
    wer = (S + I + D) / max(N, 1)
    return {"wer": wer, "accuracy": 1.0 - wer,
            "sentence_accuracy": exact / max(samples, 1), "sentences_evaluated": samples,
            "substitutions": S, "insertions": I, "deletions": D, "reference_glosses": N}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True); p.add_argument("--vocab", required=True)
    p.add_argument("--checkpoint", default="checkpoints/best.pt"); p.add_argument("--output", default="metrics.json")
    a = p.parse_args(); vocab = load_vocab(a.vocab); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LandmarkCTCBaseline(len(vocab)).to(device)
    model.load_state_dict(torch.load(a.checkpoint, map_location=device, weights_only=True))
    loader = DataLoader(GlossManifest(a.manifest, vocab), batch_size=8, collate_fn=collate)
    metrics = evaluate(model, loader, {v:k for k,v in vocab.items()}, device)
    Path(a.output).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__": main()
