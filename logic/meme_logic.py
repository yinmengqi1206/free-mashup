from __future__ import annotations

from typing import Dict, Any, List, Tuple


def generate_meme_script(
    audio_features: Dict[str, Any],
    video_features: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Generate a simple meme script based on audio and video features.
    The script is a list of actions that the video engine can execute.
    """
    script: List[Dict[str, Any]] = []

    beats: List[float] = audio_features.get("beats", [])
    motion_peaks: List[float] = video_features.get("motion_peaks", [])
    face_detected = bool(video_features.get("face_detected", False))
    target_duration = float(audio_features.get("duration", 0.0)) or float(
        video_features.get("duration", 0.0)
    )

    scenes: List[Tuple[float, float]] = video_features.get("scenes", [])
    video_duration = float(video_features.get("duration", 0.0))

    def pick_source_window(index: int, seg_len: float) -> Tuple[float, float]:
        if scenes:
            scene_start, scene_end = scenes[index % len(scenes)]
            scene_len = max(0.0, scene_end - scene_start)
            if scene_len <= seg_len:
                return scene_start, min(scene_end, scene_start + seg_len)
            offset = (index * seg_len) % (scene_len - seg_len)
            start = scene_start + offset
            return start, start + seg_len
        if video_duration <= seg_len:
            return 0.0, max(video_duration, seg_len)
        start = (index * seg_len) % (video_duration - seg_len)
        return start, start + seg_len

    def add_segment(index: int, seg_len: float) -> None:
        start, end = pick_source_window(index, seg_len)
        effects: List[Dict[str, Any]] = []
        if index % 6 == 0 and motion_peaks:
            effects.append({"op": "REVERSE"})
        if face_detected and index % 8 == 0:
            effects.append({"op": "ZOOM", "factor": 1.15})
        script.append({"op": "SEG", "start": start, "end": end, "effects": effects})

    # Build segments aligned to beats when available
    total = 0.0
    seg_len = 0.45

    if beats:
        i = 0
        while total < target_duration and i < len(beats) * 2:
            add_segment(i, seg_len)
            total += seg_len
            # Add occasional repeats for emphasis
            if i % 7 == 0:
                add_segment(i, seg_len)
                total += seg_len
            i += 1
    else:
        i = 0
        while total < target_duration and i < 200:
            add_segment(i, seg_len)
            total += seg_len
            i += 1

    # Fallback if duration is unknown or analysis produced no segments
    if not script:
        script.append({"op": "SEG", "start": 0.0, "end": min(2.0, video_duration or 2.0)})

    return script
