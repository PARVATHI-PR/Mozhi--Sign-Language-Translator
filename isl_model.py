"""Shared landmark extraction and sequence classifier for the ISL demo."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import cv2
import mediapipe as mp
import numpy as np

SEQ_LEN = 48
FEATURES = 258  # 2 x 21 hand landmarks x 3 + 33 pose landmarks x 4
_NEXT_VIDEO_TIMESTAMP_MS = 0


def make_landmarkers(model_dir: Path):
    """Create Tasks API detectors. Caller must close both objects."""
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    hand_path = model_dir / "hand_landmarker.task"
    pose_path = model_dir / "pose_landmarker_lite.task"
    if not hand_path.exists() or not pose_path.exists():
        raise FileNotFoundError(
            f"Missing landmark task files in {model_dir}. Run download_models.py first."
        )
    hand = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(hand_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.35,
            min_hand_presence_confidence=0.35,
            min_tracking_confidence=0.35,
        )
    )
    pose = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(pose_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.35,
            min_pose_presence_confidence=0.35,
            min_tracking_confidence=0.35,
        )
    )
    return hand, pose


def frame_features(rgb: np.ndarray, hand, pose, timestamp_ms: int) -> np.ndarray:
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    hr = hand.detect_for_video(image, timestamp_ms)
    pr = pose.detect_for_video(image, timestamp_ms)
    feat = np.zeros(FEATURES, dtype=np.float32)
    for i, landmarks in enumerate(hr.hand_landmarks[:2]):
        start = i * 63
        feat[start:start + 63] = np.asarray(
            [[p.x, p.y, p.z] for p in landmarks], dtype=np.float32
        ).reshape(-1)
    if pr.pose_landmarks:
        feat[126:] = np.asarray(
            [[p.x, p.y, p.z, p.visibility] for p in pr.pose_landmarks[0]],
            dtype=np.float32,
        ).reshape(-1)
    return feat


def resample(sequence: np.ndarray, length: int = SEQ_LEN) -> np.ndarray:
    if len(sequence) == 0:
        return np.zeros((length, FEATURES), dtype=np.float32)
    if len(sequence) == 1:
        return np.repeat(sequence, length, axis=0).astype(np.float32)
    old = np.linspace(0.0, 1.0, len(sequence))
    new = np.linspace(0.0, 1.0, length)
    return np.stack([np.interp(new, old, sequence[:, j]) for j in range(sequence.shape[1])], axis=1).astype(np.float32)


def video_sequence(path: Path, hand, pose, sample_fps: float = 12.0) -> np.ndarray:
    global _NEXT_VIDEO_TIMESTAMP_MS
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    stride = max(1, round(fps / sample_fps))
    frames, index = [], 0
    # Tasks VIDEO mode keeps tracker state between calls, including across clips.
    # Give each new clip a later timestamp so MediaPipe accepts it and has a gap
    # in which to discard stale tracking state.
    timestamp_base = _NEXT_VIDEO_TIMESTAMP_MS + 2000
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if index % stride == 0:
            rgb = cv2.cvtColor(cv2.resize(bgr, (640, 360)), cv2.COLOR_BGR2RGB)
            timestamp = timestamp_base + round(index * 1000 / fps)
            frames.append(frame_features(rgb, hand, pose, timestamp))
            _NEXT_VIDEO_TIMESTAMP_MS = timestamp
        index += 1
    cap.release()
    return resample(np.asarray(frames, dtype=np.float32))


def load_labels(model_path: Path) -> list[str]:
    with (model_path / "labels.json").open(encoding="utf-8") as f:
        return json.load(f)


def sentence_folders(dataset_root: Path) -> Path:
    """Accept either the corpus directory or the outer Kaggle version directory."""
    candidates = [
        dataset_root / "ISL_CSLRT_Corpus" / "ISL_CSLRT_Corpus" / "Videos_Sentence_Level",
        dataset_root / "ISL_CSLRT_Corpus" / "Videos_Sentence_Level",
        dataset_root / "Videos_Sentence_Level",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError("Could not find Videos_Sentence_Level under dataset path")
