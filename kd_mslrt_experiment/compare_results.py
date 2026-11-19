"""Compare test metrics emitted by the isolated baseline and KD experiment."""
import argparse
import json
from pathlib import Path


def read_metrics(path):
    metrics = json.loads(Path(path).read_text(encoding="utf-8"))
    if "wer" not in metrics:
        raise ValueError(f"{path} has no 'wer' metric")
    # Accuracy is derived from normalized edit distance when not logged separately.
    metrics.setdefault("accuracy", 1.0 - metrics["wer"])
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", default="../existing_baseline_model/metrics.json")
    p.add_argument("--kd", default="metrics.json")
    a = p.parse_args(); base, kd = read_metrics(a.baseline), read_metrics(a.kd)
    print(f"{'Metric':<22}{'Baseline':>14}{'KD-MSLRT':>14}{'Delta (KD-base)':>19}")
    for key in ("wer", "accuracy", "sentence_accuracy"):
        delta = kd[key] - base[key]
        print(f"{key:<22}{base[key]:>13.2%}{kd[key]:>14.2%}{delta:>18.2%}")
    print("WER delta below zero is better; accuracy delta above zero is better.")


if __name__ == "__main__": main()
