# ISL-CSLRT live phrase-to-text prototype

This trains a lightweight webcam model from the sentence-level video folders in the Kaggle ISL-CSLRT dataset. Each dataset clip is labelled with its parent folder's complete English phrase. The model therefore predicts one of the recorded phrases from a sliding webcam window; it does **not** recognize arbitrary signs or compose unseen sentences word by word.

## Windows setup

Open PowerShell in this project folder. The project-local `.venv` was created with Python 3.12 and its free dependencies have been installed. To recreate it, install Python 3.12 with PATH enabled, then run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
\.venv\Scripts\python.exe -m pip install --upgrade pip
\.venv\Scripts\python.exe -m pip install -r requirements.txt
\.venv\Scripts\python.exe download_models.py
```

## Train

```powershell
\.venv\Scripts\python.exe train_model.py --dataset 'C:\Users\PARVATHI PR\.cache\kagglehub\datasets\drblack00\isl-csltr-indian-sign-language-dataset\versions\1' --epochs 35
```

Training on the provided dataset has completed. It extracted 687 videos across 101 sentence labels and saved `artifacts\isl_gru.pt` and `artifacts\labels.json`; best held-out clip accuracy was 9.4%. This is a weak baseline, not reliable translation. The validation clips share signers with training clips, so the score does not establish generalization to new signers. To train again, this CPU extraction/training takes a while. For a quick pipeline check, use `--limit-per-class 2`; that is only a smoke-sized model and should not be presented as accurate.

## Start the webcam demo

```powershell
.\.venv\Scripts\python.exe live_translate.py
```

Press **Q** to quit and **C** to clear the accumulated output. Webcam permissions must be enabled in Windows. A phrase is appended after several recent windows agree; repeated copies of the same phrase are suppressed. The on-screen prediction is a tentative whole-phrase guess.

## Limits

The corpus has only a few clips per phrase and seven signers. The built-in validation split holds out clips, not signers, so its accuracy can overstate real-world performance. The 48-frame classifier emits whole phrase labels at a time. It is a local research/demo prototype, not a dependable interpreter. For continuous word-level translation, train a temporal gloss recognizer with aligned annotations and validate on signers excluded from training.
