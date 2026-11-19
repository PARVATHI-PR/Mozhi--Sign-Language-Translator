"""Prepare this project's ISL-CSLRT sentence clips as CTC gloss samples.

The project extractor yields 258 landmark channels. The experiment contract asks
for 276, so this adapter appends 18 explicit zero channels; change this only when
the intended 276-channel landmark mapping is defined.
"""
import argparse
import csv
import json
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from isl_model import make_landmarkers, sentence_folders, video_sequence

VIDEO_EXTS = {".mp4", ".mov", ".avi"}
_HAND = _POSE = None
_MODEL_DIR = None
_LANDMARK_FPS = 6.0
_VIDEO_FRAMES = 32
_IMAGE_SIZE = 112


def init_worker(model_dir, landmark_fps, video_frames, image_size):
    global _HAND, _POSE, _MODEL_DIR, _LANDMARK_FPS, _VIDEO_FRAMES, _IMAGE_SIZE
    cv2.setNumThreads(1)
    _MODEL_DIR = Path(model_dir); _LANDMARK_FPS = landmark_fps
    _VIDEO_FRAMES = video_frames; _IMAGE_SIZE = image_size
    _HAND, _POSE = make_landmarkers(_MODEL_DIR)


def prepare_one(task):
    i, path, label, glosses, out_dir = task
    try:
        out_dir = Path(out_dir)
        feat = video_sequence(path, _HAND, _POSE, sample_fps=_LANDMARK_FPS)
        if feat.shape[1] != 258:
            raise ValueError(f"expected 258 raw channels, got {feat.shape}")
        feat = np.pad(feat, ((0, 0), (0, 18))).astype(np.float32)
        video = sample_video(path, _VIDEO_FRAMES, _IMAGE_SIZE)
        feature_rel, video_rel = f"features/{i:04d}.npy", f"videos/{i:04d}.npy"
        np.save(out_dir / feature_rel, feat); np.save(out_dir / video_rel, video)
        return {"label": label, "features": feature_rel, "video": video_rel, "glosses": glosses}, None
    except Exception as exc:
        return None, f"{Path(path).name}: {exc}"


def load_gloss_map(dataset_root):
    corpus = dataset_root / "ISL_CSLRT_Corpus" / "ISL_CSLRT_Corpus"
    if not corpus.is_dir():
        corpus = dataset_root / "ISL_CSLRT_Corpus"
    csv_path = corpus / "corpus_csv_files" / "ISL Corpus sign glosses.csv"
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        return {r["Sentence"].strip().casefold(): r["SIGN GLOSSES"].strip().split()
                for r in csv.DictReader(f)}


def sample_video(path, frames=32, size=112):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video {path}")
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if count < 1:
        cap.release(); raise ValueError(f"No frames in {path}")
    positions = np.linspace(0, count - 1, min(frames, count)).round().astype(int)
    wanted, result, i = set(positions.tolist()), [], 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if i in wanted:
            rgb = cv2.cvtColor(cv2.resize(bgr, (size, size)), cv2.COLOR_BGR2RGB)
            result.append(np.transpose(rgb, (2, 0, 1)))
        i += 1
    cap.release()
    if not result:
        raise ValueError(f"No sampled frames in {path}")
    arr = np.asarray(result, dtype=np.uint8)
    if len(arr) < frames:
        arr = np.concatenate([arr, np.repeat(arr[-1:], frames - len(arr), axis=0)])
    return arr


def split_by_class(items, seed=42):
    rng = random.Random(seed); splits = {"train": [], "dev": [], "test": []}
    for label in sorted({item["label"] for item in items}):
        group = [x for x in items if x["label"] == label]; rng.shuffle(group)
        n = len(group)
        n_test = max(1, round(.15 * n)) if n >= 3 else 0
        n_dev = max(1, round(.15 * n)) if n >= 4 else 0
        if n - n_test - n_dev < 1:
            n_dev = 0
        splits["test"].extend(group[:n_test])
        splits["dev"].extend(group[n_test:n_test + n_dev])
        splits["train"].extend(group[n_test + n_dev:])
    return splits


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="Kaggle version folder containing ISL_CSLRT_Corpus")
    p.add_argument("--output", default="experiments/isl_cslrt_ctc")
    p.add_argument("--video-frames", type=int, default=32)
    p.add_argument("--image-size", type=int, default=112)
    p.add_argument("--landmark-fps", type=float, default=6.0)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args(); dataset = Path(a.dataset); out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    video_dir = sentence_folders(dataset); gloss_map = load_gloss_map(dataset)
    candidates = []
    for folder in sorted(p for p in video_dir.iterdir() if p.is_dir()):
        glosses = gloss_map.get(folder.name.casefold())
        if not glosses:
            print(f"No gloss annotation for folder: {folder.name}"); continue
        for path in sorted(p for p in folder.rglob("*") if p.suffix.lower() in VIDEO_EXTS):
            candidates.append({"path": path, "label": folder.name, "glosses": glosses})
    if not candidates:
        raise RuntimeError(f"No labelled videos found in {video_dir}")
    vocab_tokens = sorted({g for item in candidates for g in item["glosses"]})
    vocab = {"<blank>": 0, **{token: i + 1 for i, token in enumerate(vocab_tokens)}}
    (out / "vocab.json").write_text(json.dumps(vocab, indent=2), encoding="utf-8")
    prepared = []; tasks = []
    (out / "features").mkdir(exist_ok=True); (out / "videos").mkdir(exist_ok=True)
    tasks = [(i, str(row["path"]), row["label"], row["glosses"], str(out))
             for i, row in enumerate(candidates)]
    completed = 0
    with ProcessPoolExecutor(max_workers=max(1, a.workers), initializer=init_worker,
                             initargs=(str(Path("models").resolve()), a.landmark_fps,
                                       a.video_frames, a.image_size)) as pool:
        for result, error in pool.map(prepare_one, tasks, chunksize=1):
            completed += 1
            if result:
                prepared.append(result)
            elif error:
                print(f"Skipping {error}")
            if completed % 25 == 0 or completed == len(candidates):
                print(f"Prepared {completed}/{len(candidates)} videos ({len(prepared)} usable)", flush=True)
    split = split_by_class(prepared)
    for name, rows in split.items():
        (out / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    report = {"videos_found": len(candidates), "videos_prepared": len(prepared),
              "labels": len({r['label'] for r in prepared}), "gloss_vocab": len(vocab) - 1,
              "split_samples": {k: len(v) for k, v in split.items()},
              "landmark_channels": "258 extracted channels + 18 zero padding = 276"}
    (out / "preparation_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
