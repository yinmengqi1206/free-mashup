from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any


@dataclass
class VideoFeatures:
    scenes: List[Tuple[float, float]]
    motion_peaks: List[float]
    face_detected: bool
    duration: float


def _fallback_video_features() -> VideoFeatures:
    return VideoFeatures(scenes=[], motion_peaks=[], face_detected=False, duration=0.0)


def analyze_video(video_path: str) -> Dict[str, Any]:
    """
    Analyze video to detect scene changes, motion peaks, and faces.
    Uses opencv if available; otherwise returns a safe placeholder.
    """
    try:
        import cv2  # type: ignore
    except Exception:
        features = _fallback_video_features()
        return {
            "scenes": features.scenes,
            "motion_peaks": features.motion_peaks,
            "face_detected": features.face_detected,
            "duration": features.duration,
        }

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        features = _fallback_video_features()
        return {
            "scenes": features.scenes,
            "motion_peaks": features.motion_peaks,
            "face_detected": features.face_detected,
            "duration": features.duration,
        }

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    prev_gray = None
    frame_idx = 0
    scene_start = 0.0
    scenes: List[Tuple[float, float]] = []
    motion_peaks: List[float] = []

    # Simple face detection using Haar cascades (fast, not perfect).
    face_detected = False
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if not face_detected:
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            if len(faces) > 0:
                face_detected = True

        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray)
            motion = float(diff.mean())
            t = frame_idx / fps

            if motion > 20.0:
                motion_peaks.append(t)

            # Scene cut heuristic
            if motion > 35.0:
                scenes.append((scene_start, t))
                scene_start = t
        prev_gray = gray

    end_time = frame_idx / fps if fps > 0 else 0.0
    if end_time > scene_start:
        scenes.append((scene_start, end_time))

    cap.release()

    return {
        "scenes": scenes,
        "motion_peaks": motion_peaks,
        "face_detected": face_detected,
        "duration": end_time,
    }
