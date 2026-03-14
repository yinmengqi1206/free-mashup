from __future__ import annotations

from typing import Dict, Any, List, Any as AnyType, Tuple


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
        )
        from moviepy.video import fx as vfx  # type: ignore

        return {
            "api": "legacy",
            "VideoFileClip": VideoFileClip,
            "AudioFileClip": AudioFileClip,
            "concat_v": concatenate_videoclips,
            "concat_a": concatenate_audioclips,
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
        # Center crop
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


def apply_script(
    video_path: str,
    music_path: str | None,
    script: List[Dict[str, Any]],
    output_path: str,
) -> None:
    """
    Execute a meme script and render the output video.
    Supports CUT, REPEAT, REVERSE, ZOOM. ADD_SOUND is a placeholder.
    """
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

    segments: List[AnyType] = []
    target_size: Tuple[int, int] | None = None

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

            if target_size is None:
                try:
                    target_size = seg.size
                except Exception:
                    target_size = None
            if target_size is not None:
                seg = _fit_clip(mp, seg, target_size)

            segments.append(seg)

        elif op == "REPEAT":
            # Backward compatibility: repeat last segment
            count = int(step.get("count", 2))
            if segments:
                segments.extend([segments[-1]] * (count - 1))

        elif op == "REVERSE":
            if segments:
                seg = segments.pop()
                if mp["api"] == "legacy":
                    seg = seg.fx(mp["vfx"].all.time_mirror)
                else:
                    seg = mp["TimeMirror"]().apply(seg)
                segments.append(seg)

        elif op == "ZOOM":
            factor = float(step.get("factor", 1.2))
            if segments:
                seg = segments.pop()
                if mp["api"] == "legacy":
                    seg = seg.resize(factor)
                else:
                    seg = mp["Resize"](factor).apply(seg)
                segments.append(seg)

        elif op == "ADD_SOUND":
            # Placeholder. Use moviepy CompositeAudioClip for real overlay.
            pass

        elif op == "ADD_IMAGE":
            # Placeholder for future image overlays.
            pass

        elif op == "ADD_TEXT":
            # Placeholder for future text overlays.
            pass

    if not segments:
        segments = [get_source(video_path)]

    final = mp["concat_v"](segments, method="compose")

    if music:
        target = float(getattr(final, "duration", 0.0))
        if target > 0:
            if music.duration < target:
                # Loop audio by concatenation, then trim.
                loops = int(target // music.duration) + 1
                music = mp["concat_a"]([music] * loops)
            if mp["api"] == "legacy":
                music = music.subclip(0, target)
                final = final.set_audio(music)
            else:
                music = music.subclipped(0, target)
                final = final.with_audio(music)

    final.write_videofile(output_path, codec="libx264", audio_codec="aac")

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
