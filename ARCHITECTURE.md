# Prototype architecture

## Current prototype and final plan

The table distinguishes the implemented prototype from the stated direction for a fuller system. The final-plan column is based on the recommendation in `README_MODEL.md`; details beyond that recommendation are goals to define, not implemented features.

| Area | Current prototype | Final plan / target system |
| --- | --- | --- |
| Recognition unit | Classifies a 48-frame window as one of 101 complete phrases represented in the dataset. | Recognize a continuous stream of signs at word level, using temporal gloss recognition. |
| Language output | Selects a stored English phrase label; does not assemble unseen sentences. | Convert recognized gloss sequences into continuous word-level text, including combinations not seen as complete training phrases. |
| Training data | Sentence-level clips with phrase-folder labels; current baseline uses 687 clips. | Use data with aligned gloss or word-level annotations suitable for temporal recognition. |
| Model pipeline | MediaPipe hand and pose landmarks feed a bidirectional GRU phrase classifier. | Replace or extend the phrase classifier with a temporal gloss recognizer; the exact model and decoding/language stages remain to be designed. |
| Evaluation | Clip-level holdout; clips may share signers with training. Recorded accuracy is 9.4%. | Validate with signers excluded from training and assess continuous word-level translation quality. |
| Intended use | Local experimental webcam demo; predictions are tentative and may be unreliable. | Target a more general and dependable continuous translation system, subject to data, evaluation, and usability results. |

The final-plan column describes a direction, not a completed or guaranteed capability. The available project notes do not specify a final model architecture, interface, or acceptance metrics.

## Purpose and scope

This is a local, webcam based phrase classifier for the ISL-CSLRT sentence level video corpus. It predicts one complete English phrase from a rolling 48 frame window. It does not identify word or gloss boundaries, compose new sentences, or translate arbitrary signing.

## Components

| Component | Responsibility |
| --- | --- |
| `dataset_import.py` | Optional KaggleHub entry point for downloading the corpus. |
| `download_models.py` | Downloads the MediaPipe hand and pose Tasks model files into `models/`. |
| `isl_model.py` | Shared feature extraction, video sampling, resampling, label loading, and dataset path discovery. |
| `train_model.py` | Finds labelled sentence clips, extracts features, trains and validates the sequence classifier, and writes model artifacts. |
| `live_translate.py` | Captures webcam frames, runs live inference, smooths predictions, and displays/appends phrase output. |
| `models/` | MediaPipe task model files used to detect hand and pose landmarks. |
| `artifacts/` | Trained PyTorch checkpoint (`isl_gru.pt`) and ordered phrase labels (`labels.json`). |

## Data and training flow

1. The dataset is arranged in sentence-label folders. `sentence_folders()` accepts the corpus directory or supported outer Kaggle version directory layouts.
2. `train_model.py` sorts folder names to create stable class IDs and gathers video paths beneath each phrase folder.
3. Videos are sampled at approximately 12 frames per second. Frames are resized to 640×360 and converted to RGB.
4. MediaPipe VIDEO mode detects up to two hands and one pose. Each sampled frame becomes 258 float features: 126 values for two 21-point hand landmark sets (x/y/z), and 132 values for 33 pose landmarks (x/y/z/visibility). Missing detections remain zero-filled.
5. Each clip's feature sequence is linearly resampled to 48 frames. The shared VIDEO-mode detectors are closed after extraction.
6. Features and class IDs are split into training and validation clips, stratified by class where possible. A two-layer bidirectional GRU classifier is trained using PyTorch; the best validation checkpoint and phrase label list are saved under `artifacts/`.

## Live inference flow

1. `live_translate.py` loads the checkpoint and label list, constructs the same `SentenceGRU` architecture, and initializes the MediaPipe hand and pose detectors.
2. OpenCV reads the camera (device 0, DirectShow backend) and samples at 12 frames per second, matching training. Each frame passes through the shared 258-feature extractor.
3. Once the rolling deque contains 48 feature vectors, the model produces a probability distribution over the complete phrase labels.
4. The most likely phrase and confidence are shown on screen. A six-window history stabilizes output; a phrase is appended after it wins at least four windows, meets the mean-score threshold, differs from the last emitted phrase, and the one-second cooldown has elapsed.
5. Press **C** to clear accumulated phrase output and prediction history; press **Q** to exit. Camera, landmarkers, and OpenCV windows are released on exit.

## Runtime dependencies

Python dependencies are listed in `requirements.txt`: MediaPipe, OpenCV, NumPy, and PyTorch. The two MediaPipe `.task` files are downloaded separately by `download_models.py`. Training also needs access to the corpus; the dataset itself is not included in the runtime artifacts.

## Known constraints

- The classifier operates at whole phrase level; output is a sequence of phrase guesses, not continuous word level translation.
- The corpus has few examples per phrase and a small number of signers. Validation holds out clips rather than signers, so its score does not establish performance on unseen signers.
- The recorded baseline in `README_MODEL.md` reached 9.4% held-out clip accuracy across 101 phrase labels and is not reliable for practical interpretation.
- Webcam output is experimental. Landmark visibility, camera setup, signing variation, and repeated or overlapping motion can affect predictions.

## Main interfaces

- Train: `python train_model.py --dataset <dataset-version-directory> [--epochs 35] [--limit-per-class N] [--output artifacts]`
- Prepare detector models: `python download_models.py`
- Run webcam demo: `python live_translate.py`

See `README_MODEL.md` for the Windows setup steps and the dataset-specific training example.
