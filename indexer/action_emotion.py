from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass
class ActionEmotion:
    action: str
    action_score: float
    emotion: str
    emotion_score: float


def analyze_action_emotion(_frames_bgr: np.ndarray) -> ActionEmotion:
    """
    Placeholder for action/emotion models.
    If optional models are installed, you can implement real inference here.
    """
    # Default fallback
    return ActionEmotion(
        action="unknown",
        action_score=0.0,
        emotion="unknown",
        emotion_score=0.0,
    )
