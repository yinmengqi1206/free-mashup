from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any


@dataclass
class AudioFeatures:
    bpm: float
    beats: List[float]
    speech_segments: List[Tuple[float, float]]
    duration: float


def _fallback_audio_features() -> AudioFeatures:
    # Minimal placeholder when audio analysis deps are unavailable.
    return AudioFeatures(bpm=120.0, beats=[], speech_segments=[], duration=0.0)


def analyze_audio(audio_path: str) -> Dict[str, Any]:
    """
    Analyze audio to extract bpm, beats, and speech segments.
    Uses librosa if available; otherwise returns a safe placeholder.
    """
    try:
        import librosa  # type: ignore
    except Exception:
        features = _fallback_audio_features()
        return {
            "bpm": features.bpm,
            "beats": features.beats,
            "speech_segments": features.speech_segments,
            "duration": features.duration,
        }

    # Load audio
    y, sr = librosa.load(audio_path, sr=None, mono=True)

    # Tempo and beat tracking
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    # Placeholder for speech segments; hook whisper or other ASR here.
    speech_segments: List[Tuple[float, float]] = []
    duration = float(librosa.get_duration(y=y, sr=sr))

    return {
        "bpm": float(tempo),
        "beats": [float(t) for t in beat_times],
        "speech_segments": speech_segments,
        "duration": duration,
    }
