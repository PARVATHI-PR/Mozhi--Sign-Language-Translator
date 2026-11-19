# KD-MSLRT test report

**Run date:** 2026-10-04  
**Teacher:** local 3D-CNN/BiLSTM scaffold, trained 5 epochs on CPU; best dev CTC loss 4.6172.  
**Student:** 5 epochs on CPU; best dev total loss 31.7360 (epoch 3).  
**Split:** same ISL-CSLRT clip split and gloss vocabulary as the CTC baseline.

## Held-out test results

| Metric | Result |
| --- | ---: |
| WER | 100.00% |
| Derived gloss accuracy (`1 - WER`) | 0.00% |
| Exact sentence accuracy | 0.00% |
| Sentences | 98 |
| Reference glosses | 349 |
| Substitutions / insertions / deletions | 0 / 0 / 349 |

The greedy CTC decoder emitted only blanks for every test clip. A diagnostic evaluation of the teacher also produced 100.00% WER with 349 deletions, so this teacher did not provide useful gloss predictions in the short run. The student result is therefore a failed pilot, not evidence against a fully trained KD-MSLRT model.

## Reproduction

```powershell
python kd_mslrt_experiment/train_teacher.py --train kd_mslrt_experiment/_run_data/train.jsonl --dev kd_mslrt_experiment/_run_data/dev.jsonl --vocab kd_mslrt_experiment/_run_data/vocab.json --epochs 5 --batch-size 4 --output kd_mslrt_experiment/checkpoints/teacher.pt
python kd_mslrt_experiment/train_kd.py --train kd_mslrt_experiment/_run_data/train.jsonl --dev kd_mslrt_experiment/_run_data/dev.jsonl --vocab kd_mslrt_experiment/_run_data/vocab.json --teacher kd_mslrt_experiment/checkpoints/teacher.pt --epochs 5 --output kd_mslrt_experiment/checkpoints/best.pt
python kd_mslrt_experiment/evaluate.py --manifest kd_mslrt_experiment/_run_data/test.jsonl --vocab kd_mslrt_experiment/_run_data/vocab.json --checkpoint kd_mslrt_experiment/checkpoints/best.pt --output kd_mslrt_experiment/metrics.json
```

## Limits

This is not an exact paper reproduction: the teacher is a locally implemented 3D-CNN/BiLSTM scaffold rather than CorrNet; the available extractor's 258 channels are padded with 18 zeros to reach 276; the split is by clips rather than signer; and training was shortened to 5 epochs instead of 80. These limitations, especially the collapsed teacher, make the KD WER unsuitable as a quality estimate for the published method.
