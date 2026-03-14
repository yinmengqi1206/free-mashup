from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any


@dataclass
class AudioFeatures:
    bpm: float
    beats: List[float]
    speech_segments: List[Tuple[float, float]]
    duration: float
    rms: List[float]
    spectral_centroid: List[float]
    chroma: List[List[float]]


def _fallback_audio_features() -> AudioFeatures:
    # Minimal placeholder when audio analysis deps are unavailable.
    return AudioFeatures(
        bpm=120.0,
        beats=[],
        speech_segments=[],
        duration=0.0,
        rms=[],
        spectral_centroid=[],
        chroma=[],
    )


def analyze_audio(audio_path: str) -> Dict[str, Any]:
    """
    Analyze audio to extract bpm, beats, and rich features per short frame.
    Uses librosa if available; otherwise returns a safe placeholder.
    """
    try:
        import librosa  # type: ignore
        import numpy as np  # type: ignore
    except Exception:
        features = _fallback_audio_features()
        try:
            from moviepy import AudioFileClip  # type: ignore
            with AudioFileClip(audio_path) as clip:
                features.duration = float(clip.duration)
        except Exception:
            pass
        return {
            "bpm": features.bpm,
            "beats": features.beats,
            "speech_segments": features.speech_segments,
            "duration": features.duration,
            "rms": features.rms,
            "spectral_centroid": features.spectral_centroid,
            "chroma": features.chroma,
        }

    # Load audio
    y, sr = librosa.load(audio_path, sr=None, mono=True)

    # Tempo and beat tracking
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    # Frame-level features
    rms = librosa.feature.rms(y=y)[0]
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)

    # Align to time for simpler matching
    frame_times = librosa.frames_to_time(range(len(rms)), sr=sr)
    duration = float(librosa.get_duration(y=y, sr=sr))

    return {
        "bpm": float(tempo),
        "beats": [float(t) for t in beat_times],
        "speech_segments": [],
        "duration": duration,
        "rms": [float(v) for v in rms],
        "spectral_centroid": [float(v) for v in spectral_centroid],
        "chroma": [[float(x) for x in row] for row in chroma],
        "frame_times": [float(t) for t in frame_times],
    }
