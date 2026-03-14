from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class ClipVisualFeatures:
    duration: float
    motion_mean: float
    motion_curve: List[float]
    brightness_mean: float
    brightness_curve: List[float]
    color_mean: Tuple[float, float, float]
    color_std: Tuple[float, float, float]
    embedding: np.ndarray


def compute_visual_features(
    frames: np.ndarray,
    fps: float,
) -> Tuple[List[float], List[float], Tuple[float, float, float], Tuple[float, float, float]]:
    # frames: (N, H, W, 3) in BGR
    if frames.size == 0:
        return [], [], (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)

    # Brightness curve
    gray = frames.mean(axis=3)
    brightness_curve = gray.mean(axis=(1, 2)).astype(float).tolist()
    brightness_mean = float(np.mean(brightness_curve))

    # Motion curve
    motion_curve = []
    prev = None
    for f in gray:
        if prev is not None:
            motion_curve.append(float(np.mean(np.abs(f - prev))))
        prev = f
    if not motion_curve:
        motion_curve = [0.0] * max(1, len(brightness_curve) - 1)
    motion_mean = float(np.mean(motion_curve))

    # Color stats (BGR)
    color_mean = tuple(float(x) for x in frames.reshape(-1, 3).mean(axis=0))
    color_std = tuple(float(x) for x in frames.reshape(-1, 3).std(axis=0))

    return motion_curve, brightness_curve, color_mean, color_std
