from __future__ import annotations

import random
from typing import Dict, Any, List, Tuple

import numpy as np

from indexer.vector_db import ClipRow, list_clips


def _build_intervals(beats: List[float], target_duration: float) -> List[float]:
    if len(beats) >= 2:
        intervals = [max(0.2, beats[i + 1] - beats[i]) for i in range(len(beats) - 1)]
        total = sum(intervals)
        if total < target_duration:
            median = sorted(intervals)[len(intervals) // 2]
            while total < target_duration:
                intervals.append(median)
                total += median
        return intervals
    step = 0.45
    count = int(max(1, target_duration // step))
    return [step] * count


def _duration_score(clip: ClipRow, seg_len: float) -> float:
    if seg_len <= 0:
        return 0.0
    return 1.0 - min(abs(clip.duration - seg_len) / seg_len, 1.0)


def _audio_window_stats(
    frame_times: List[float],
    feature: List[float],
    start: float,
    end: float,
) -> float:
    if not frame_times or not feature:
        return 0.0
    total = 0.0
    count = 0
    for t, v in zip(frame_times, feature):
        if t < start:
            continue
        if t > end:
            break
        total += v
        count += 1
    return total / max(count, 1)


def _audio_window_vector(
    frame_times: List[float],
    chroma: List[List[float]],
    start: float,
    end: float,
) -> np.ndarray:
    if not frame_times or not chroma:
        return np.zeros(12, dtype=np.float32)
    total = np.zeros(12, dtype=np.float32)
    count = 0
    for idx, t in enumerate(frame_times):
        if t < start:
            continue
        if t > end:
            break
        col = np.asarray([row[idx] for row in chroma], dtype=np.float32)
        total += col
        count += 1
    if count == 0:
        return total
    total /= float(count)
    return total


def _kmeans(x: np.ndarray, k: int, iters: int = 12) -> Tuple[np.ndarray, np.ndarray]:
    n = x.shape[0]
    if n == 0:
        return np.zeros((0, x.shape[1])), np.zeros((0,), dtype=np.int64)
    k = max(1, min(k, n))
    # Initialize with random samples
    idx = np.random.choice(n, size=k, replace=False)
    centroids = x[idx].copy()
    labels = np.zeros(n, dtype=np.int64)
    for _ in range(iters):
        # Assign
        dists = np.linalg.norm(x[:, None, :] - centroids[None, :, :], axis=2)
        labels = np.argmin(dists, axis=1)
        # Update
        for j in range(k):
            members = x[labels == j]
            if len(members) == 0:
                continue
            centroids[j] = members.mean(axis=0)
    return centroids, labels


def _build_audio_segments(
    audio_features: Dict[str, Any],
    intervals: List[float],
) -> List[Dict[str, Any]]:
    frame_times = audio_features.get("frame_times", [])
    rms = audio_features.get("rms", [])
    centroid = audio_features.get("spectral_centroid", [])
    chroma = audio_features.get("chroma", [])

    max_rms = max(rms) if rms else 1.0
    max_centroid = max(centroid) if centroid else 1.0

    segments = []
    t = 0.0
    for seg_len in intervals:
        start_t = t
        end_t = t + seg_len
        t = end_t

        r = _audio_window_stats(frame_times, rms, start_t, end_t)
        c = _audio_window_stats(frame_times, centroid, start_t, end_t)
        ch = _audio_window_vector(frame_times, chroma, start_t, end_t)

        r_norm = r / max_rms if max_rms > 0 else 0.0
        c_norm = c / max_centroid if max_centroid > 0 else 0.0
        ch_norm = ch / (np.linalg.norm(ch) + 1e-8)

        feat = np.concatenate([
            np.asarray([r_norm, c_norm], dtype=np.float32),
            ch_norm.astype(np.float32),
        ])
        segments.append({
            "start": start_t,
            "end": end_t,
            "len": seg_len,
            "rms": r_norm,
            "centroid": c_norm,
            "feature": feat,
        })
    return segments


def _build_clip_features(clips: List[ClipRow]) -> np.ndarray:
    feats = []
    motion_vals = [c.motion for c in clips]
    bright_vals = [c.brightness for c in clips]
    motion_min, motion_max = min(motion_vals), max(motion_vals)
    bright_min, bright_max = min(bright_vals), max(bright_vals)
    motion_range = max(motion_max - motion_min, 1.0)
    bright_range = max(bright_max - bright_min, 1.0)

    for c in clips:
        motion_norm = (c.motion - motion_min) / motion_range
        bright_norm = (c.brightness - bright_min) / bright_range
        colorfulness = float(np.linalg.norm(np.asarray(c.color_std)) / 255.0)
        feats.append([motion_norm, bright_norm, colorfulness])
    return np.asarray(feats, dtype=np.float32)


def generate_meme_script_from_db(
    audio_features: Dict[str, Any],
    db_path: str,
    min_seg: float = 0.25,
    max_seg: float = 0.9,
) -> List[Dict[str, Any]]:
    """
    Uniqueness-first, category-driven strategy:
    1) Cluster audio segments into melody/energy categories.
    2) Cluster clips into visual categories.
    3) Map audio clusters to clip clusters (energy -> motion).
    4) Use token-bucket per cluster (no replacement), refill only when empty.
    """
    beats: List[float] = audio_features.get("beats", [])
    target_duration = float(audio_features.get("duration", 0.0)) or 0.0
    if target_duration <= 0:
        target_duration = beats[-1] if beats else 10.0

    intervals = _build_intervals(beats, target_duration)
    all_clips = list_clips(db_path)
    if not all_clips:
        raise ValueError("No clips available in vector database.")

    segments = _build_audio_segments(audio_features, intervals)
    seg_k = min(8, max(2, len(segments) // 8))
    clip_k = seg_k

    # Cluster audio segments
    seg_features = np.asarray([s["feature"] for s in segments], dtype=np.float32)
    seg_centroids, seg_labels = _kmeans(seg_features, seg_k)

    # Cluster clips by visual features
    clip_features = _build_clip_features(all_clips)
    clip_centroids, clip_labels = _kmeans(clip_features, clip_k)

    # Map audio clusters to clip clusters by energy->motion ranking
    seg_energy = []
    for k in range(seg_k):
        idxs = np.where(seg_labels == k)[0]
        avg_rms = float(np.mean([segments[i]["rms"] for i in idxs])) if len(idxs) else 0.0
        avg_centroid = float(np.mean([segments[i]["centroid"] for i in idxs])) if len(idxs) else 0.0
        seg_energy.append((k, avg_rms + 0.5 * avg_centroid))
    seg_energy.sort(key=lambda x: x[1])

    clip_motion = []
    for k in range(clip_k):
        idxs = np.where(clip_labels == k)[0]
        avg_motion = float(np.mean([all_clips[i].motion for i in idxs])) if len(idxs) else 0.0
        clip_motion.append((k, avg_motion))
    clip_motion.sort(key=lambda x: x[1])

    cluster_map = {}
    for (s_k, _), (c_k, _) in zip(seg_energy, clip_motion):
        cluster_map[s_k] = c_k

    # Build token buckets per clip cluster
    buckets: Dict[int, List[ClipRow]] = {k: [] for k in range(clip_k)}
    for idx, c in enumerate(all_clips):
        buckets[clip_labels[idx]].append(c)

    for k in buckets:
        random.shuffle(buckets[k])

    script: List[Dict[str, Any]] = []

    for i, seg in enumerate(segments):
        seg_len = max(min_seg, min(max_seg, seg["len"]))
        seg_label = int(seg_labels[i])
        clip_cluster = cluster_map.get(seg_label, 0)

        # Token-bucket: no replacement within cluster
        if not buckets[clip_cluster]:
            # Refill from full cluster (allow reuse after exhaustion)
            buckets[clip_cluster] = [c for idx, c in enumerate(all_clips) if clip_labels[idx] == clip_cluster]
            random.shuffle(buckets[clip_cluster])

        # Pick best duration match from bucket
        best = None
        best_score = -1.0
        best_idx = -1
        for idx, c in enumerate(buckets[clip_cluster]):
            s = _duration_score(c, seg_len)
            if s > best_score:
                best_score = s
                best = c
                best_idx = idx

        if best is None:
            best = random.choice(buckets[clip_cluster])
            best_idx = buckets[clip_cluster].index(best)

        # Remove token
        buckets[clip_cluster].pop(best_idx)

        if best.duration > seg_len:
            start = best.start + random.random() * max(0.01, best.duration - seg_len)
            end = start + seg_len
        else:
            start = best.start
            end = best.end

        script.append(
            {
                "op": "SEG",
                "video_path": best.video_path,
                "start": float(start),
                "end": float(end),
                "effects": [],
            }
        )

    return script
