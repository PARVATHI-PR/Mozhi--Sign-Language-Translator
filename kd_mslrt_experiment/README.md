# KD-MSLRT experiment

This folder implements the requested 3D-video-to-1D-landmark distillation experiment without changing the root phrase-level webcam baseline. The available project corpus has complete English phrase labels, not frame-aligned gloss sequences, so it cannot train/evaluate this CTC experiment as-is.

## Data and checkpoints required

Use JSONL manifests with one row per sample: `features` is a relative `.npy` float32 array `[T,276]`, `video` is a relative `.npy` RGB float32 array `[T,3,H,W]` scaled to `[0,1]`, and `glosses` is an ordered list of gloss tokens. Use shared train/dev/test partitions and `vocab.json` across both experiments; `<blank>` must have ID 0.

The teacher module is a heavy 3D-CNN/BiLSTM teacher with the output contract required for KD. It is a CorrNet-style implementation, not the authors' exact CorrNet source or pretrained weights. `train_teacher.py` trains this local teacher scaffold on the prepared corpus before `train_kd.py`; it does not reproduce the paper's CorrNet teacher.

`prepare_data.py` converts the cached project corpus into manifests, a gloss vocabulary, 276-channel inputs and sampled videos. It maps sentence folders through the corpus gloss CSV. The shared extractor provides 258 values, so preprocessing appends 18 zero channels to meet the requested shape. This adapter and the teacher scaffold differ from the paper's exact landmarks and CorrNet; treat results as a local pipeline experiment, not an exact reproduction. Keep generated data and checkpoints local because sampled videos take substantial disk space.

## Implemented recipe

- Four 1D convolution layers, kernel size 5, max pooling, BiLSTM, and shared gloss classifier on 276-D inputs.
- Conv-map, BiLSTM-map, and student self-distillation KL terms, plus CTC loss.
- Frozen teacher during student optimization. Default optimizer recipe is Adam, weight decay `1e-4`, initial LR `1e-4`, linear decay, 80 epochs.
- Training loss uses `CTC + 25 * (Lc + Lb + Ls)` to implement the supplied 1:25 CTC:KD training ratio. Each KD term uses the paper's alpha/temperature multiplier (default 1.0).
- Landmark augmentation probabilities: 30% rotation (up to 30 degrees), 50% translation with a 50% horizontal-reflection subchance, and 20% adjacent-frame fusion.
- Evaluation uses the same greedy CTC WER implementation as the baseline.

## Run

```powershell
python kd_mslrt_experiment/prepare_data.py --dataset 'C:\Users\PARVATHI PR\.cache\kagglehub\datasets\drblack00\isl-csltr-indian-sign-language-dataset\versions\1' --output kd_mslrt_experiment/_run_data
python existing_baseline_model/train.py --train kd_mslrt_experiment/_run_data/train.jsonl --dev kd_mslrt_experiment/_run_data/dev.jsonl --vocab kd_mslrt_experiment/_run_data/vocab.json --output existing_baseline_model/checkpoints/best.pt
python existing_baseline_model/evaluate.py --manifest kd_mslrt_experiment/_run_data/test.jsonl --vocab kd_mslrt_experiment/_run_data/vocab.json --checkpoint existing_baseline_model/checkpoints/best.pt --output existing_baseline_model/metrics.json
python kd_mslrt_experiment/train_teacher.py --train kd_mslrt_experiment/_run_data/train.jsonl --dev kd_mslrt_experiment/_run_data/dev.jsonl --vocab kd_mslrt_experiment/_run_data/vocab.json --output kd_mslrt_experiment/checkpoints/teacher.pt
python kd_mslrt_experiment/train_kd.py --train kd_mslrt_experiment/_run_data/train.jsonl --dev kd_mslrt_experiment/_run_data/dev.jsonl --vocab kd_mslrt_experiment/_run_data/vocab.json --teacher kd_mslrt_experiment/checkpoints/teacher.pt --output kd_mslrt_experiment/checkpoints/best.pt
python kd_mslrt_experiment/evaluate.py --manifest kd_mslrt_experiment/_run_data/test.jsonl --vocab kd_mslrt_experiment/_run_data/vocab.json --checkpoint kd_mslrt_experiment/checkpoints/best.pt --output kd_mslrt_experiment/metrics.json
python kd_mslrt_experiment/compare_results.py --baseline existing_baseline_model/metrics.json --kd kd_mslrt_experiment/metrics.json
```

The KD loss aligns teacher probability maps to the student's pooled time axis by linear interpolation. Horizontal reflection negates x coordinates; left/right landmark indices are not swapped because no index mapping was supplied.
