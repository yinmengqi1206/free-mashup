from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, List

from analysis.audio_analysis import analyze_audio
from analysis.video_analysis import analyze_video
from logic.meme_logic import generate_meme_script
from engine.video_engine import apply_script, EngineUnavailable


ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "input"
OUTPUT_DIR = ROOT / "output"


def build_media_index(input_dir: Path) -> Dict[str, List[str]]:
    videos = sorted(str(p) for p in (input_dir / "videos").glob("*"))
    audios = sorted(str(p) for p in (input_dir / "audios").glob("*"))
    images = sorted(str(p) for p in (input_dir / "images").glob("*"))

    return {"videos": videos, "audios": audios, "images": images}


def save_index(index: Dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")


def pick_first(paths: List[str]) -> str | None:
    return paths[0] if paths else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Auto meme video generator")
    parser.add_argument("--video", help="Path to a source video")
    parser.add_argument("--music", help="Path to a background music file")
    parser.add_argument("--output", help="Output video path")

    args = parser.parse_args()

    index = build_media_index(INPUT_DIR)
    save_index(index, ROOT / "media_index.json")

    video_path = args.video or pick_first(index["videos"])
    music_path = args.music or pick_first(index["audios"])

    if not video_path:
        print("No video found. Put files into input/videos/")
        return 1

    audio_features = (
        analyze_audio(music_path)
        if music_path
        else {"bpm": 120.0, "beats": [], "speech_segments": [], "duration": 0.0}
    )
    video_features = analyze_video(video_path)

    # If no audio duration is available, fall back to video length.
    if not audio_features.get("duration"):
        audio_features["duration"] = video_features.get("duration", 0.0)

    script = generate_meme_script(audio_features, video_features)

    output_path = args.output or str(OUTPUT_DIR / "meme_001.mp4")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        apply_script(video_path, music_path, script, output_path)
    except EngineUnavailable as exc:
        print(str(exc))
        return 2

    print(f"Rendered: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
