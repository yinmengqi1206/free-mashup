from __future__ import annotations

import datetime as dt
import logging
import os
import time
from dataclasses import dataclass
from typing import Iterator, List, Tuple

import numpy as np

from indexer.action_emotion import analyze_action_emotion
from indexer.embedder import ClipEmbedder
from indexer.vector_db import (
    add_clips,
    delete_clips_for_video,
    init_db,
    upsert_video,
    video_needs_processing,
)
from indexer.video_features import compute_visual_features

LOGGER = logging.getLogger(__name__)


@dataclass
class ClipMeta:
    start: float
    end: float
    duration: float
    motion: float
    brightness: float
    action: str
    action_score: float
    emotion: str
    emotion_score: float
    color_mean: Tuple[float, float, float]
    color_std: Tuple[float, float, float]
    motion_curve: List[float]
    brightness_curve: List[float]
    embedding: np.ndarray


def _iter_videos(video_dir: str) -> Iterator[str]:
    for root, _, files in os.walk(video_dir):
        for name in files:
            if name.lower().endswith((".mp4", ".mov", ".mkv", ".avi")):
                yield os.path.join(root, name)


def _detect_scenes(
    video_path: str,
    threshold: float = 35.0,
    min_len: float = 0.4,
    frame_step: int = 1,
) -> List[Tuple[float, float]]:
    try:
        import cv2  # type: ignore
    except Exception:
        return [(0.0, 0.0)]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [(0.0, 0.0)]

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    prev_gray = None
    scenes: List[Tuple[float, float]] = []
    scene_start = 0.0
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray)
            motion = float(diff.mean())
            t = frame_idx / fps
            if motion > threshold:
                if t - scene_start >= min_len:
                    scenes.append((scene_start, t))
                scene_start = t
        prev_gray = gray

        # Skip frames for faster scanning on very large videos
        if frame_step > 1:
            for _ in range(frame_step - 1):
                if not cap.grab():
                    break
            frame_idx += frame_step - 1

    end_time = frame_idx / fps if fps > 0 else 0.0
    if end_time - scene_start >= min_len:
        scenes.append((scene_start, end_time))

    cap.release()
    return scenes or [(0.0, end_time)]


def _sample_frames(video_path: str, start: float, end: float, samples: int = 12) -> Tuple[np.ndarray, float]:
    import cv2  # type: ignore

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    duration = max(0.01, end - start)
    for i in range(samples):
        t = start + duration * (i + 0.5) / samples
        frame_idx = int(t * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()
    if not frames:
        return np.zeros((0, 0, 3), dtype=np.uint8), fps
    return np.stack(frames, axis=0), fps


def _build_clips(video_path: str, scenes: List[Tuple[float, float]], embedder: ClipEmbedder) -> List[ClipMeta]:
    clips: List[ClipMeta] = []
    last_log = time.time()
    log_every = int(os.environ.get("SCAN_LOG_EVERY", "10"))
    log_every = max(1, log_every)
    samples = int(os.environ.get("SCAN_SAMPLES", "12"))
    samples = max(1, samples)
    for idx, (start, end) in enumerate(scenes):
        duration = max(0.0, end - start)
        if duration <= 0.2:
            continue
        frames, fps = _sample_frames(video_path, start, end, samples=samples)
        # Use mid frame for embedding
        mid_idx = frames.shape[0] // 2 if frames.size else 0
        mid_frame = frames[mid_idx] if frames.size else np.zeros((224, 224, 3), dtype=np.uint8)
        embedding = embedder.embed_image(mid_frame)
        motion_curve, brightness_curve, color_mean, color_std = compute_visual_features(frames, fps)
        motion = float(np.mean(motion_curve)) if motion_curve else 0.0
        brightness = float(np.mean(brightness_curve)) if brightness_curve else 0.0
        action_emotion = analyze_action_emotion(frames)
        clips.append(
            ClipMeta(
                start=float(start),
                end=float(end),
                duration=float(duration),
                motion=float(motion),
                brightness=float(brightness),
                action=action_emotion.action,
                action_score=action_emotion.action_score,
                emotion=action_emotion.emotion,
                emotion_score=action_emotion.emotion_score,
                color_mean=color_mean,
                color_std=color_std,
                motion_curve=motion_curve,
                brightness_curve=brightness_curve,
                embedding=embedding,
            )
        )
        if idx % log_every == 0 and idx > 0:
            now = time.time()
            LOGGER.info(
                "Processed %s clips for %s (%.1fs since last log)",
                idx + 1,
                os.path.basename(video_path),
                now - last_log,
            )
            last_log = now
    return clips


def scan_folder(video_dir: str, db_path: str) -> None:
    init_db(db_path)
    embedder = ClipEmbedder()

    try:
        import cv2  # type: ignore  # noqa: F401
    except Exception:
        LOGGER.error("opencv-python is required for scanning. Install dependencies first.")
        return

    frame_step = int(os.environ.get("SCAN_STRIDE", "3"))
    frame_step = max(1, frame_step)

    for video_path in _iter_videos(video_dir):
        stat = os.stat(video_path)
        if not video_needs_processing(db_path, video_path, stat.st_mtime, stat.st_size):
            LOGGER.info("Skip unchanged video: %s", video_path)
            continue

        # Basic metadata for large files
        try:
            import cv2  # type: ignore
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
            duration = frames / fps if fps else 0.0
            cap.release()
        except Exception:
            fps = 0.0
            duration = 0.0

        size_gb = stat.st_size / (1024 ** 3)
        LOGGER.info(
            "Scanning video: %s (%.2f GB, %.1f s, stride=%s)",
            video_path,
            size_gb,
            duration,
            frame_step,
        )
        scenes = _detect_scenes(video_path, frame_step=frame_step)
        LOGGER.info("Detected %s scenes", len(scenes))

        delete_clips_for_video(db_path, video_path)
        start_ts = time.time()
        clips = _build_clips(video_path, scenes, embedder)
        LOGGER.info("Feature extraction done in %.1fs", time.time() - start_ts)
        add_clips(
            db_path,
            (
                (
                    video_path,
                    c.start,
                    c.end,
                    c.duration,
                    c.motion,
                    c.brightness,
                    c.action,
                    c.action_score,
                    c.emotion,
                    c.emotion_score,
                    c.color_mean,
                    c.color_std,
                    c.embedding,
                    np.asarray(c.motion_curve, dtype=np.float32),
                    np.asarray(c.brightness_curve, dtype=np.float32),
                )
                for c in clips
            ),
        )
        upsert_video(
            db_path,
            video_path,
            stat.st_mtime,
            stat.st_size,
            dt.datetime.utcnow().isoformat(),
        )
        LOGGER.info("Indexed %s clips for %s", len(clips), video_path)

    LOGGER.info("Scan completed")
