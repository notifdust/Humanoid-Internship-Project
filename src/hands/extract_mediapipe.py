"""Extract right-hand landmarks (MediaPipe Tasks HandLandmarker) from 1080p ego clips."""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from tqdm import tqdm

WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


def ensure_model(model_path: Path) -> Path:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if model_path.exists() and model_path.stat().st_size > 0:
        return model_path
    print(f"Downloading HandLandmarker model -> {model_path}")
    urllib.request.urlretrieve(MODEL_URL, model_path)
    return model_path


def lm_to_xyz(lm) -> tuple[float, float, float]:
    return float(lm.x), float(lm.y), float(lm.z)


def _unflip_x(pt: tuple[float, float, float]) -> tuple[float, float, float]:
    # Detection runs on a horizontally flipped frame (helps ego cams).
    return (1.0 - pt[0], pt[1], pt[2])


def extract_clip(path: Path, out_json: Path, landmarker: vision.HandLandmarker) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    frames = []
    detected = 0
    total = 0

    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        total += 1
        # Ego: physical right hand often looks like a mirrored left hand.
        rgb = cv2.cvtColor(cv2.flip(bgr, 1), cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        # VIDEO mode needs monotonically increasing timestamps (ms)
        ts_ms = int(round((total - 1) * 1000.0 / fps))
        result = landmarker.detect_for_video(mp_image, ts_ms)

        rec = {
            "frame": total - 1,
            "t": (total - 1) / fps,
            "detected": False,
            "handedness": None,
            "wrist": None,
            "index_tip": None,
            "thumb_tip": None,
            "middle_tip": None,
            "landmarks": None,
        }

        if result.hand_landmarks:
            choice = 0
            # After flip, prefer MediaPipe "Right" (= physical right in ego).
            if result.handedness:
                for i, hands in enumerate(result.handedness):
                    label = hands[0].category_name if hands else ""
                    if label.lower() == "right":
                        choice = i
                        break
                if result.handedness[choice]:
                    rec["handedness"] = result.handedness[choice][0].category_name

            lm = result.hand_landmarks[choice]
            pts = [_unflip_x(lm_to_xyz(p)) for p in lm]
            rec["detected"] = True
            rec["landmarks"] = pts
            rec["wrist"] = pts[WRIST]
            rec["index_tip"] = pts[INDEX_TIP]
            rec["thumb_tip"] = pts[THUMB_TIP]
            rec["middle_tip"] = pts[MIDDLE_TIP]
            detected += 1

        frames.append(rec)

    cap.release()
    visibility = detected / total if total else 0.0
    summary = {
        "filename": path.name,
        "width": w,
        "height": h,
        "fps": fps,
        "num_frames": total,
        "detected_frames": detected,
        "visibility": round(visibility, 4),
        "qa_pass": visibility >= 0.80,
        "frames": frames,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f)
    return {
        "filename": path.name,
        "visibility": summary["visibility"],
        "qa_pass": summary["qa_pass"],
        "num_frames": total,
        "detected_frames": detected,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=Path("data/processed/ego_1080"))
    parser.add_argument("--dst", type=Path, default=Path("data/processed/hands"))
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/hand_landmarker.task"),
    )
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    clips = sorted(args.src.glob("pick_place_mug_*.mp4"))
    if args.limit > 0:
        clips = clips[: args.limit]
    if not clips:
        raise SystemExit(f"No 1080p clips in {args.src}. Run resize_videos.py first.")

    ensure_model(args.model)

    def make_landmarker() -> vision.HandLandmarker:
        base_options = mp_python.BaseOptions(model_asset_path=str(args.model))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3,
            min_tracking_confidence=0.3,
        )
        return vision.HandLandmarker.create_from_options(options)

    report = []
    for src in tqdm(clips, desc="mediapipe hands"):
        out = args.dst / (src.stem + ".json")
        # Always re-extract (ego flip tweak); delete old JSON first.
        if out.exists():
            out.unlink()
        # Fresh landmarker per clip so video timestamps restart at 0.
        with make_landmarker() as landmarker:
            report.append(extract_clip(src, out, landmarker))

    qa_path = args.dst / "qa_report.json"
    args.dst.mkdir(parents=True, exist_ok=True)
    with qa_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    passed = sum(1 for r in report if r.get("qa_pass"))
    print(f"QA pass (≥80% hand visible): {passed}/{len(report)}")
    print(f"Report: {qa_path}")


if __name__ == "__main__":
    main()
