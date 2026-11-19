"""Download the small official MediaPipe Tasks landmark models."""
from pathlib import Path
from urllib.request import urlretrieve

MODELS = {
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "pose_landmarker_lite.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
}

if __name__ == "__main__":
    dest = Path("models")
    dest.mkdir(exist_ok=True)
    for name, url in MODELS.items():
        path = dest / name
        if path.exists():
            print(f"Already present: {path}")
            continue
        print(f"Downloading {name}...")
        urlretrieve(url, path)
    print("Landmarker models are ready.")
