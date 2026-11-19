# CTC baseline vs KD student

**Run date:** 2026-10-04  
Both systems used the same 98 test clips and 167-gloss vocabulary.

| Metric | CTC baseline | KD student | KD minus baseline |
| --- | ---: | ---: | ---: |
| WER (lower is better) | 88.83% | 100.00% | +11.17 percentage points |
| Derived gloss accuracy (`1 - WER`) | 11.17% | 0.00% | -11.17 percentage points |
| Exact sentence accuracy | 0.00% | 0.00% | 0.00 percentage points |
| Sentences evaluated | 98 | 98 | — |

## Interpretation

The KD student performed worse in this pilot and emitted blank output for every test example. The teacher also had 100% WER, so the distillation source had collapsed to blank predictions. Baseline training ran for 10 epochs; teacher and student training ran for 5 epochs each, all on CPU. This is a pipeline smoke-scale comparison, not a comparison with the paper's 80-epoch training or published results.

The result is further limited by the input adapter: the project's 258 extracted values were padded with 18 zeros to meet the required 276-channel model input. The train/dev/test split was randomized by clip, not signer. The previous root phrase classifier's 9.4% phrase accuracy is excluded because it measures a different task.

Metrics are saved as JSON alongside these reports: `existing_baseline_model/metrics.json`, `kd_mslrt_experiment/teacher_metrics.json`, and `kd_mslrt_experiment/metrics.json`. Rerun both training pipelines with a valid 276-channel mapping, signer-independent split, and longer training before drawing conclusions about KD-MSLRT.
