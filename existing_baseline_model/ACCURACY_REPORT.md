# CTC baseline test report

**Run date:** 2026-10-04  
**Checkpoint:** `checkpoints/best.pt`  
**Training:** 10 epochs, CPU, best dev CTC loss 4.4206.  
**Split:** shared ISL-CSLRT clip split; 485 train, 97 dev, 98 test; 167 glosses.

## Held-out test results

| Metric | Result |
| --- | ---: |
| WER | 88.83% |
| Derived gloss accuracy (`1 - WER`) | 11.17% |
| Exact sentence accuracy | 0.00% |
| Sentences | 98 |
| Reference glosses | 349 |
| Substitutions / insertions / deletions | 59 / 0 / 251 |

The accuracy column is derived from WER; it is not frame accuracy. Exact sentence accuracy requires the entire predicted gloss sequence to match.

## Reproduction

```powershell
python existing_baseline_model/train.py --train kd_mslrt_experiment/_run_data/train.jsonl --dev kd_mslrt_experiment/_run_data/dev.jsonl --vocab kd_mslrt_experiment/_run_data/vocab.json --epochs 10 --output existing_baseline_model/checkpoints/best.pt
python existing_baseline_model/evaluate.py --manifest kd_mslrt_experiment/_run_data/test.jsonl --vocab kd_mslrt_experiment/_run_data/vocab.json --checkpoint existing_baseline_model/checkpoints/best.pt --output existing_baseline_model/metrics.json
```

This is a short CPU pilot, not the paper's 80-epoch recipe. Its input arrays use the existing 258-channel MediaPipe extractor padded with 18 zeros to satisfy the requested 276-dimension interface. The randomized clip split is not signer-disjoint. The root phrase classifier's previously recorded 9.4% phrase accuracy is a different task and is not compared here.
