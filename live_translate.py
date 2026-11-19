"""Live webcam phrase recognizer for the trained ISL-CSLRT phrase model."""
from collections import Counter, deque
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from isl_model import SEQ_LEN, FEATURES, frame_features, load_labels, make_landmarkers, resample
from train_model import SentenceGRU


def main():
    model_dir = Path("artifacts")
    checkpoint = torch.load(model_dir / "isl_gru.pt", map_location="cpu", weights_only=False)
    labels = load_labels(model_dir)
    model = SentenceGRU(len(labels))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    hand, pose = make_landmarkers(Path("models"))
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Close apps currently using the camera and try again.")
    history = deque(maxlen=SEQ_LEN)
    recent_predictions = deque(maxlen=6)
    sentence = []
    last_emit = 0.0
    last_sample = 0.0
    last_emitted_phrase = ""
    prediction, confidence = "Show a sign", 0.0
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            now = time.monotonic()
            # Training samples clips at 12 fps; match that rate for webcam windows.
            if now - last_sample >= 1.0 / 12.0:
                rgb = cv2.cvtColor(cv2.resize(bgr, (640, 360)), cv2.COLOR_BGR2RGB)
                history.append(frame_features(rgb, hand, pose, int(now * 1000)))
                last_sample = now
                if len(history) == SEQ_LEN:
                    with torch.no_grad():
                        probs = torch.softmax(model(torch.from_numpy(np.asarray(history)[None])), dim=1)[0]
                    confidence, idx = probs.max(0)
                    confidence = float(confidence)
                    prediction = labels[int(idx)]
                    recent_predictions.append((prediction, confidence))
                    counts = Counter(label for label, _ in recent_predictions)
                    stable_phrase, stable_count = counts.most_common(1)[0]
                    stable_scores = [score for label, score in recent_predictions if label == stable_phrase]
                    if (len(recent_predictions) == recent_predictions.maxlen
                            and stable_count >= 4
                            and float(np.mean(stable_scores)) >= 0.10
                            and stable_phrase != last_emitted_phrase
                            and now - last_emit >= 1.0):
                        sentence.append(stable_phrase)
                        last_emitted_phrase = stable_phrase
                        last_emit = now
            canvas = cv2.resize(bgr, (960, 540))
            cv2.putText(canvas, f"Phrase: {prediction[:70]} ({confidence:.0%})", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (40, 240, 60), 2)
            cv2.putText(canvas, "Live text (phrase-level): " + " | ".join(sentence)[-90:], (20, 510), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.imshow("ISL live phrase-to-text | Q quit | C clear", canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                sentence.clear()
                last_emitted_phrase = ""
                recent_predictions.clear()
                last_emit = time.monotonic()
    finally:
        cap.release()
        hand.close()
        pose.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
