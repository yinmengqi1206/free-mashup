from __future__ import annotations

import logging
import os
from typing import Dict, Any, List, Any as AnyType, Tuple

import numpy as np


LOGGER = logging.getLogger(__name__)


class EngineUnavailable(Exception):
    pass


def _require_moviepy():
    try:
        # Legacy MoviePy (pre-2.0) API
        from moviepy.editor import (  # type: ignore
            VideoFileClip,
            AudioFileClip,
            concatenate_videoclips,
            concatenate_audioclips,
            TextClip,
            CompositeVideoClip,
            ImageClip,
        )
        from moviepy.video import fx as vfx  # type: ignore

        return {
            "api": "legacy",
            "VideoFileClip": VideoFileClip,
            "AudioFileClip": AudioFileClip,
            "concat_v": concatenate_videoclips,
            "concat_a": concatenate_audioclips,
            "TextClip": TextClip,
            "CompositeVideoClip": CompositeVideoClip,
            "ImageClip": ImageClip,
            "vfx": vfx,
        }
    except Exception:
        try:
            # MoviePy 2.x API
            from moviepy import (  # type: ignore
                VideoFileClip,
                AudioFileClip,
                concatenate_videoclips,
                concatenate_audioclips,
                TextClip,
                CompositeVideoClip,
                ImageClip,
            )
            from moviepy.video.fx.TimeMirror import TimeMirror  # type: ignore
            from moviepy.video.fx.Loop import Loop  # type: ignore
            from moviepy.video.fx.Resize import Resize  # type: ignore
            from moviepy.video.fx.Crop import Crop  # type: ignore

            return {
                "api": "modern",
                "VideoFileClip": VideoFileClip,
                "AudioFileClip": AudioFileClip,
                "concat_v": concatenate_videoclips,
                "concat_a": concatenate_audioclips,
                "TextClip": TextClip,
                "CompositeVideoClip": CompositeVideoClip,
                "ImageClip": ImageClip,
                "TimeMirror": TimeMirror,
                "Loop": Loop,
                "Resize": Resize,
                "Crop": Crop,
            }
        except Exception as exc:
            raise EngineUnavailable(
                "moviepy is required for rendering. Install dependencies from requirements.txt"
            ) from exc


def _fit_clip(mp: Dict[str, Any], clip: AnyType, target_size: Tuple[int, int]) -> AnyType:
    tw, th = target_size
    if tw <= 0 or th <= 0:
        return clip
    try:
        w, h = clip.size
    except Exception:
        return clip

    # Scale to cover target (no black bars), then center-crop
    scale = max(tw / float(w), th / float(h))
    if mp["api"] == "legacy":
        clip = clip.resize(scale)
        x_center = clip.size[0] / 2.0
        y_center = clip.size[1] / 2.0
        clip = clip.fx(
            mp["vfx"].crop,
            x_center=x_center,
            y_center=y_center,
            width=tw,
            height=th,
        )
    else:
        clip = mp["Resize"](scale).apply(clip)
        x_center = clip.size[0] / 2.0
        y_center = clip.size[1] / 2.0
        clip = mp["Crop"](x_center=x_center, y_center=y_center, width=tw, height=th).apply(clip)
    return clip


def _render_text_image(text: str, font_size: int, max_w: int) -> AnyType:
    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore
    except Exception:
        return None

    font_size = int(font_size)
    max_w = int(max_w)

    font_path = os.environ.get("SUBTITLE_FONT", "")
    font = None
    if font_path and os.path.exists(font_path):
        try:
            font = ImageFont.truetype(font_path, font_size)
        except Exception:
            font = None

    if font is None:
        # Common macOS Chinese font
        mac_font = "/System/Library/Fonts/PingFang.ttc"
        if os.path.exists(mac_font):
            try:
                font = ImageFont.truetype(mac_font, font_size)
            except Exception:
                font = None

    if font is None:
        font = ImageFont.load_default()

    # Create a temporary image to measure text
    pad_x = 12
    pad_y = 20
    img = Image.new("RGBA", (max_w, int(font_size * 5.0)), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    text_bbox = draw.multiline_textbbox((0, 0), text, font=font, align="center")
    # Add extra padding to avoid clipping descenders and long lines
    w = int(min(max_w, text_bbox[2] - text_bbox[0] + pad_x * 2))
    h = int(text_bbox[3] - text_bbox[1] + pad_y * 2)
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    # Warm gold for cinematic lyrics
    draw.multiline_text(
        (pad_x, pad_y),
        text,
        font=font,
        fill=(255, 220, 140, 255),
        align="center",
        stroke_width=2,
        stroke_fill=(0, 0, 0, 220),
    )
    return img


def apply_script(
    video_path: str,
    music_path: str | None,
    script: List[Dict[str, Any]],
    output_path: str,
    subtitles: List[Dict[str, Any]] | None = None,
    subtitle_side: str = "right",
    quality_preset: str = "high",
) -> None:
    mp = _require_moviepy()
    VideoFileClip = mp["VideoFileClip"]
    AudioFileClip = mp["AudioFileClip"]

    sources: Dict[str, AnyType] = {}

    def get_source(path: str):
        if path not in sources:
            sources[path] = VideoFileClip(path)
        return sources[path]

    music = None
    if music_path:
        try:
            music = AudioFileClip(music_path)
        except Exception:
            music = None

    segments_raw: List[AnyType] = []
    max_w = 0
    max_h = 0

    for step in script:
        op = step.get("op")

        if op in {"SEG", "CUT"}:
            source_path = step.get("video_path") or video_path
            source = get_source(source_path)
            start = float(step.get("start", 0.0))
            end = float(step.get("end", start + 0.3))
            max_end = max(0.0, float(getattr(source, "duration", 0.0)))
            start = max(0.0, min(start, max_end))
            end = max(0.0, min(end, max_end))
            if end <= start:
                continue
            if mp["api"] == "legacy":
                seg = source.subclip(start, end)
            else:
                seg = source.subclipped(start, end)

            for effect in step.get("effects", []):
                eop = effect.get("op")
                if eop == "REVERSE":
                    if mp["api"] == "legacy":
                        seg = seg.fx(mp["vfx"].all.time_mirror)
                    else:
                        seg = mp["TimeMirror"]().apply(seg)
                elif eop == "ZOOM":
                    factor = float(effect.get("factor", 1.2))
                    if mp["api"] == "legacy":
                        seg = seg.resize(factor)
                    else:
                        seg = mp["Resize"](factor).apply(seg)

            try:
                w, h = seg.size
                max_w = max(max_w, int(w))
                max_h = max(max_h, int(h))
            except Exception:
                pass

            segments_raw.append(seg)

    if not segments_raw:
        segments_raw = [get_source(video_path)]

    target_size = (max_w, max_h) if max_w > 0 and max_h > 0 else None
    segments: List[AnyType] = []
    if target_size is None:
        segments = segments_raw
    else:
        for seg in segments_raw:
            segments.append(_fit_clip(mp, seg, target_size))

    final = mp["concat_v"](segments, method="compose")

    if music:
        target = float(getattr(final, "duration", 0.0))
        if target > 0:
            if music.duration < target:
                loops = int(target // music.duration) + 1
                music = mp["concat_a"]([music] * loops)
            if mp["api"] == "legacy":
                music = music.subclip(0, target)
                final = final.set_audio(music)
            else:
                music = music.subclipped(0, target)
                final = final.with_audio(music)

    # Subtitles (LRC) rendered via PIL to avoid ImageMagick dependency
    if subtitles:
        try:
            ImageClip = mp["ImageClip"]
            CompositeVideoClip = mp["CompositeVideoClip"]
            w, h = final.size
            # Centered, upper-third placement with larger type
            x_pos = int(w * 0.5)
            y_pos = int(h * (1.0 / 6.0))
            max_w = int(w * 0.90)
            font_size = int(max(32, int(h * 0.055)))
            overlays = [final]
            LOGGER.info(
                "Subtitle overlay: lines=%s, size=%sx%s, font=%s, pos=(center,%s)",
                len(subtitles),
                w,
                h,
                font_size,
                y_pos,
            )
            skipped = 0
            target = float(getattr(final, "duration", 0.0))
            for s in subtitles:
                try:
                    img = _render_text_image(s["text"], font_size, max_w)
                    if img is None:
                        LOGGER.warning("Subtitle render returned None for text: %s", s.get("text"))
                        continue
                    txt = ImageClip(np.array(img))
                    start_t = float(s["start"])
                    end_t = float(s["end"])
                    if target > 0 and start_t >= target:
                        skipped += 1
                        continue
                    if target > 0:
                        end_t = min(end_t, target)
                    dur = max(0.1, end_t - start_t)
                    if mp["api"] == "legacy":
                        txt = txt.set_position(("center", y_pos))
                        txt = txt.set_start(start_t).set_duration(dur)
                    else:
                        txt = txt.with_position(("center", y_pos))
                        txt = txt.with_start(start_t).with_duration(dur)
                    overlays.append(txt)
                except Exception as exc:
                    LOGGER.warning("Subtitle line failed: %s (%s)", s.get("text"), exc)
            if skipped:
                LOGGER.warning("Skipped %s subtitle lines beyond video duration %.2fs", skipped, target)
            final = CompositeVideoClip(overlays, size=final.size)
        except Exception as exc:
            LOGGER.warning("Failed to render subtitles: %s", exc)

    ffmpeg_params = ["-crf", "18", "-preset", "slow"] if quality_preset == "high" else ["-crf", "22", "-preset", "medium"]
    final.write_videofile(
        output_path,
        codec="libx264",
        audio_codec="aac",
        ffmpeg_params=ffmpeg_params,
        bitrate="8000k" if quality_preset == "high" else "5000k",
    )

    for clip in sources.values():
        try:
            clip.close()
        except Exception:
            pass
    if music is not None:
        try:
            music.close()
        except Exception:
            pass
