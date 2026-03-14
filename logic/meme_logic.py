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

    # Diversity controls
    recent_window = max(6, min(24, len(segments) // 6))
    base_cooldown = max(8, min(32, len(segments) // 4))
    usage: Dict[int, int] = {}
    cooldown_until: Dict[int, int] = {}
    recent: List[int] = []

    # Precompute neighbor clusters by energy proximity
    cluster_order = [c_k for _, c_k in clip_motion]
    cluster_index = {c_k: i for i, c_k in enumerate(cluster_order)}

    def candidate_pool(seg_label: int, seg_len: float) -> List[ClipRow]:
        # Primary cluster
        primary = cluster_map.get(seg_label, 0)
        pools = []
        if buckets.get(primary):
            pools.append(buckets[primary])

        # Neighboring clusters for small pools
        idx = cluster_index.get(primary, 0)
        for offset in (1, -1, 2, -2):
            j = idx + offset
            if 0 <= j < len(cluster_order):
                c_k = cluster_order[j]
                if buckets.get(c_k):
                    pools.append(buckets[c_k])

        # Flatten
        flat = [c for p in pools for c in p]
        if not flat:
            flat = [c for c in all_clips if _duration_score(c, seg_len) > 0]
        return flat or all_clips

    def pick_clip(pool: List[ClipRow], seg_len: float, step: int) -> ClipRow:
        # Filter by cooldown and recent history
        filtered = []
        for c in pool:
            if c.clip_id in recent:
                continue
            if cooldown_until.get(c.clip_id, -1) > step:
                continue
            filtered.append(c)
        if not filtered:
            filtered = pool

        # Sort by duration match, then take top K
        scored = [(c, _duration_score(c, seg_len)) for c in filtered]
        scored.sort(key=lambda x: x[1], reverse=True)
        top_k = max(10, min(40, len(scored) // 4))
        top = scored[:top_k] if scored else []

        # Prefer lower usage counts for balancing
        def pick_weight(item: Tuple[ClipRow, float]) -> float:
            c, s = item
            u = usage.get(c.clip_id, 0)
            return (s + 1e-6) / (1.0 + u)

        if top:
            weights = [pick_weight(x) for x in top]
            return random.choices([c for c, _ in top], weights=weights, k=1)[0]

        return random.choice(filtered)

    for i, seg in enumerate(segments):
        seg_len = max(min_seg, min(max_seg, seg["len"]))
        seg_label = int(seg_labels[i])

        pool = candidate_pool(seg_label, seg_len)
        clip = pick_clip(pool, seg_len, i)

        # Token-bucket removal to maximize uniqueness
        for k, bucket in buckets.items():
            for idx, c in enumerate(bucket):
                if c.clip_id == clip.clip_id:
                    bucket.pop(idx)
                    break

        if clip.duration > seg_len:
            start = clip.start + random.random() * max(0.01, clip.duration - seg_len)
            end = start + seg_len
        else:
            start = clip.start
            end = clip.end

        script.append(
            {
                "op": "SEG",
                "video_path": clip.video_path,
                "start": float(start),
                "end": float(end),
                "effects": [],
            }
        )

        # Update diversity trackers
        usage[clip.clip_id] = usage.get(clip.clip_id, 0) + 1
        cooldown_until[clip.clip_id] = i + base_cooldown + usage[clip.clip_id]
        recent.append(clip.clip_id)
        if len(recent) > recent_window:
            recent.pop(0)

    return script
