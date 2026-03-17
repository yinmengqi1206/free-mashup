from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple

from analysis.audio_analysis import analyze_audio
from engine.video_engine import apply_script, EngineUnavailable
from indexer.scanner import scan_folder
from logic.meme_logic import generate_meme_script_from_db


ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "input"
OUTPUT_DIR = ROOT / "output"
INDEX_DIR = ROOT / "index"
DEFAULT_DB = INDEX_DIR / "clip_db.sqlite"


def build_media_index(input_dir: Path) -> Dict[str, List[str]]:
    videos = sorted(str(p) for p in (input_dir / "videos").glob("*"))
    audios = sorted(str(p) for p in (input_dir / "audios").glob("*"))
    images = sorted(str(p) for p in (input_dir / "images").glob("*"))

    return {"videos": videos, "audios": audios, "images": images}


def save_index(index: Dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")


def pick_first(paths: List[str]) -> str | None:
    return paths[0] if paths else None


def _load_lyrics(path: str) -> List[Dict[str, Any]]:
    """
    Load .lrc lyrics file into a list of {start, end, text}.
    If the format is unknown, return empty list.
    """
    if not path or not os.path.exists(path):
        return []
    lines = Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()
    entries: List[Tuple[float, str]] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("[") and "]" in line:
            tag, text = line.split("]", 1)
            tag = tag.strip("[")
            parts = tag.split(":")
            if len(parts) >= 2:
                try:
                    minutes = int(parts[0])
                    seconds = float(parts[1])
                    t = minutes * 60 + seconds
                    text = text.strip()
                    if text:
                        entries.append((t, text))
                except ValueError:
                    continue
    entries.sort(key=lambda x: x[0])
    results: List[Dict[str, Any]] = []
    for i, (t, text) in enumerate(entries):
        end = entries[i + 1][0] if i + 1 < len(entries) else t + 3.0
        results.append({"start": t, "end": end, "text": text})
    return results


def _setup_logging(verbose: bool) -> None:
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def cmd_scan(args: argparse.Namespace) -> int:
    _setup_logging(True)
    scan_folder(args.videos, args.db)
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    _setup_logging(args.verbose)

    index = build_media_index(INPUT_DIR)
    save_index(index, ROOT / "media_index.json")

    music_path = args.audio or pick_first(index["audios"])
    if not music_path:
        print("No audio found. Put files into input/audios/")
        return 1

    audio_features = analyze_audio(music_path)

    try:
        script = generate_meme_script_from_db(audio_features, args.db)
    except Exception as exc:
        print(f"Failed to build script. {exc}")
        return 1

    output_path = args.output or str(OUTPUT_DIR / "meme_001.mp4")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Use the first segment's video as fallback.
    fallback_video = script[0].get("video_path") if script else None
    if not fallback_video:
        print("No video found in script. Run scan first.")
        return 1

    lyrics_path = args.lyrics
    if not lyrics_path:
        default_lyrics = INPUT_DIR / "audios" / "lyrics.lrc"
        if default_lyrics.exists():
            lyrics_path = str(default_lyrics)

    try:
        subtitles = _load_lyrics(lyrics_path) if lyrics_path else []
        if subtitles:
            logging.getLogger(__name__).info("Loaded %s subtitle lines from %s", len(subtitles), lyrics_path)
            logging.getLogger(__name__).info("First subtitle: %s", subtitles[0])
        else:
            logging.getLogger(__name__).warning("No subtitles loaded (check LRC format/path)")
        apply_script(
            fallback_video,
            music_path,
            script,
            output_path,
            subtitles=subtitles,
            subtitle_side="center",
            quality_preset=args.quality,
        )
    except EngineUnavailable as exc:
        print(str(exc))
        return 2

    print(f"Rendered: {output_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Auto meme video generator")
    subparsers = parser.add_subparsers(dest="command", required=False)

    scan_parser = subparsers.add_parser("scan", help="Scan videos and build vector DB")
    scan_parser.add_argument("--videos", default=str(INPUT_DIR / "videos"), help="Video folder")
    scan_parser.add_argument("--db", default=str(DEFAULT_DB), help="Vector DB path")
    scan_parser.add_argument("--verbose", action="store_true", help="Verbose logs")
    scan_parser.set_defaults(func=cmd_scan)

    gen_parser = subparsers.add_parser("generate", help="Generate meme video from audio")
    gen_parser.add_argument("--audio", help="Audio file path")
    gen_parser.add_argument("--db", default=str(DEFAULT_DB), help="Vector DB path")
    gen_parser.add_argument("--output", help="Output video path")
    gen_parser.add_argument("--verbose", action="store_true", help="Verbose logs")
    gen_parser.add_argument("--lyrics", help="Lyrics file (.lrc) for subtitles")
    gen_parser.add_argument("--quality", choices=["high", "medium"], default="high", help="Render quality preset")
    gen_parser.set_defaults(func=cmd_generate)

    args = parser.parse_args()

    if getattr(args, "func", None):
        return args.func(args)

    # Default behavior: generate using default inputs
    args.command = "generate"
    args.db = str(DEFAULT_DB)
    args.output = None
    args.audio = None
    args.verbose = False
    return cmd_generate(args)


if __name__ == "__main__":
    raise SystemExit(main())
