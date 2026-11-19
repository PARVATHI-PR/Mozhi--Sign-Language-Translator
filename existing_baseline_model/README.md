# Isolated gloss CTC baseline

This is a CTC baseline used only for comparison with `kd_mslrt_experiment`. It is separate from the root phrase-classification prototype. It requires frame-level landmarks and gloss sequence annotations; the existing sentence-folder dataset is not sufficient.

## Data format

Create `vocab.json` with `{"<blank>": 0, "GLOSS_A": 1, ...}` and a JSON Lines manifest for each split. Each row has `features` (a relative `.npy` path containing float32 `[frames,276]`) and `glosses` (a list of vocabulary strings in order). Use the same train/dev/test samples and vocabulary for baseline and KD runs. Gloss arrays must not contain `<blank>`.

## Run

From this folder, with the project Python environment active:

```powershell
python train.py --train data/train.jsonl --dev data/dev.jsonl --vocab data/vocab.json
python evaluate.py --manifest data/test.jsonl --vocab data/vocab.json --checkpoint checkpoints/best.pt
```

Evaluation writes `metrics.json` with greedy CTC WER and edit counts. `accuracy` is the derived `1 - WER` and is not a sentence exact-match rate.
