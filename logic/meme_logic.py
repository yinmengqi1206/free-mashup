from __future__ import annotations

import random
from typing import Dict, Any, List, Tuple

import numpy as np

from indexer.vector_db import ClipRow, list_clips


def _normalize(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-8)


def _smooth(x: List[float], win: int = 8) -> List[float]:
    if not x:
        return []
    out = []
    for i in range(len(x)):
        start = max(0, i - win)
        end = min(len(x), i + win + 1)
        out.append(sum(x[start:end]) / max(1, end - start))
    return out


def _music_sections(frame_times: List[float], rms: List[float]) -> List[Tuple[float, float, str]]:
    if not frame_times or not rms:
        return [(0.0, 9999.0, "build")]

    smooth = _smooth(rms, win=12)
    mean = float(np.mean(smooth))
    std = float(np.std(smooth))

    labels = []
    for v in smooth:
        if v < mean - 0.3 * std:
            labels.append("intro")
        elif v < mean + 0.2 * std:
            labels.append("build")
        elif v < mean + 0.7 * std:
            labels.append("climax")
        else:
            labels.append("climax")

    sections: List[Tuple[float, float, str]] = []
    current = labels[0]
    start_t = frame_times[0]
    for i in range(1, len(labels)):
        if labels[i] != current:
            end_t = frame_times[i]
            sections.append((start_t, end_t, current))
            start_t = frame_times[i]
            current = labels[i]
    sections.append((start_t, frame_times[-1], current))
    return sections


def _section_for_time(sections: List[Tuple[float, float, str]], t: float) -> str:
    for start, end, label in sections:
        if start <= t <= end:
            return label
    return sections[-1][2] if sections else "build"


def _segment_length(label: str) -> float:
    if label == "intro":
        return random.uniform(4.0, 7.0)
    if label == "climax":
        return random.uniform(2.0, 3.2)
    if label == "outro":
        return random.uniform(4.0, 7.0)
    return random.uniform(3.0, 5.0)


def _build_segments(audio_features: Dict[str, Any]) -> List[Dict[str, Any]]:
    frame_times = audio_features.get("frame_times", [])
    rms = audio_features.get("rms", [])
    duration = float(audio_features.get("duration", 0.0) or 0.0)
    if duration <= 0:
        duration = frame_times[-1] if frame_times else 60.0

    sections = _music_sections(frame_times, rms)

    segments = []
    t = 0.0
    while t < duration:
        label = _section_for_time(sections, t)
        seg_len = _segment_length(label)
        seg_len = max(1.8, min(seg_len, 8.0))
        if t + seg_len > duration:
            seg_len = duration - t
        segments.append({"start": t, "end": t + seg_len, "len": seg_len, "label": label})
        t += seg_len
    return segments


def _motion_energy_targets(audio_features: Dict[str, Any], segments: List[Dict[str, Any]]) -> List[float]:
    frame_times = audio_features.get("frame_times", [])
    rms = audio_features.get("rms", [])
    if not frame_times or not rms:
        return [0.5 for _ in segments]

    max_rms = max(rms) if rms else 1.0
    targets = []
    for seg in segments:
        total = 0.0
        count = 0
        for t, v in zip(frame_times, rms):
            if t < seg["start"]:
                continue
            if t > seg["end"]:
                break
            total += v
            count += 1
        avg = total / max(1, count)
        targets.append(min(avg / max_rms, 1.0))
    return targets


def generate_meme_script_from_db(
    audio_features: Dict[str, Any],
    db_path: str,
    min_seg: float = 1.8,
    max_seg: float = 8.0,
) -> List[Dict[str, Any]]:
    """
    Cinematic montage strategy:
    - Long shots (2.5-6s) with section-aware pacing
    - Movie grouping (soft limit 3-4 shots)
    - Strong diversity across movies
    - Energy -> motion matching
    """
    all_clips = list_clips(db_path)
    if not all_clips:
        raise ValueError("No clips available in vector database.")

    segments = _build_segments(audio_features)
    energy_targets = _motion_energy_targets(audio_features, segments)

    # Normalize motion for scoring
    motion_vals = [c.motion for c in all_clips]
    motion_min, motion_max = min(motion_vals), max(motion_vals)
    motion_range = max(motion_max - motion_min, 1.0)

    # Usage tracking
    used_clip_ids: set[int] = set()
    recent: List[int] = []
    recent_window = 8

    video_use: Dict[str, int] = {}
    for c in all_clips:
        video_use.setdefault(c.video_path, 0)

    current_movie = None
    current_run = 0
    max_run = 4

    script: List[Dict[str, Any]] = []

    for i, seg in enumerate(segments):
        seg_len = max(min_seg, min(max_seg, seg["len"]))
        target_energy = energy_targets[i]
        target_motion = motion_min + target_energy * motion_range

        # Candidate pool: allow slightly shorter clips for diversity
        pool = [
            c
            for c in all_clips
            if c.clip_id not in used_clip_ids and c.duration >= seg_len * 0.6
        ]
        if not pool:
            pool = [c for c in all_clips if c.duration >= seg_len * 0.6]
            used_clip_ids.clear()

        # If some videos are underused, focus on them
        min_use = min(video_use.values()) if video_use else 0
        max_use = max(video_use.values()) if video_use else 0
        if max_use - min_use >= 2:
            underused = {v for v, u in video_use.items() if u == min_use}
            under_pool = [c for c in pool if c.video_path in underused]
            if under_pool:
                pool = under_pool

        # Score: duration match + motion match + diversity + continuity
        scored: List[Tuple[float, ClipRow]] = []
        avg_use = (sum(video_use.values()) / max(len(video_use), 1)) if video_use else 0.0
        for c in pool:
            if c.clip_id in recent:
                continue
            duration_score = 1.0 - min(abs(c.duration - seg_len) / seg_len, 1.0)
            motion_score = 1.0 - min(abs(c.motion - target_motion) / motion_range, 1.0)
            diversity_bonus = 0.12 if video_use.get(c.video_path, 0) == 0 else 0.0
            usage_penalty = 0.08 * (video_use.get(c.video_path, 0) / max(avg_use + 1.0, 1.0))
            continuity_bonus = 0.08 if current_movie and c.video_path == current_movie else 0.0
            score = 0.55 * duration_score + 0.35 * motion_score + diversity_bonus + continuity_bonus - usage_penalty
            scored.append((score, c))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_k = max(10, min(30, len(scored) // 4))
        top = [c for _, c in scored[:top_k]] if scored else []

        # Enforce soft movie run limit
        if current_movie and current_run >= max_run:
            top = [c for c in top if c.video_path != current_movie] or top

        if top:
            clip = random.choice(top)
        else:
            clip = random.choice(pool)

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

        used_clip_ids.add(clip.clip_id)
        recent.append(clip.clip_id)
        if len(recent) > recent_window:
            recent.pop(0)

        video_use[clip.video_path] = video_use.get(clip.video_path, 0) + 1

        if current_movie == clip.video_path:
            current_run += 1
        else:
            current_movie = clip.video_path
            current_run = 1

    return script
