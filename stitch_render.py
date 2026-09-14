"""
Maxis Film Stitch — single-job runner (replaces AWS Lambda + the unused
EC2 poll-loop design in server/stitch_worker.py in the Maxis repo).

Dispatched on demand by the Browsight dashboard (app.py's
/api/render/generate, mirroring api_clip_generate's pattern) with
--video-id/--supabase-url/--supabase-key passed per call — same
credential-per-request convention every other Browsight job already uses,
so nothing here needs standing Supabase credentials of its own.

Does, for exactly one video:
  1. Downloads each approved scene (video clip or reference image)
  2. Builds each scene's audio (voiceover / native clip audio / bg music)
  3. FFmpeg-concatenates all scene segments into one final MP4
  4. Saves the final video to this VPS's local media storage (no AWS)
  5. Updates the video row: rendered_video_url + status = 'render_complete'

Logic is otherwise identical to server/stitch_worker.py — see that file's
original docstring for the full audio-priority/captions/QC rationale.
"""

import argparse
import asyncio
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import edge_tts
import requests

ROOT = Path(__file__).parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [stitch] %(levelname)s %(message)s",
)
log = logging.getLogger("stitch_render")

EDGE_TTS_FALLBACK_VOICE = "en-US-AriaNeural"


# ── Supabase (plain REST, no client library — matches clip_worker.py) ──────────

class SupabaseVideos:
    def __init__(self, url: str, key: str):
        self._url = url.rstrip("/")
        self._headers = {
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }

    def get(self, video_id: str) -> dict | None:
        r = requests.get(
            f"{self._url}/rest/v1/videos?id=eq.{video_id}&select=*",
            headers=self._headers, timeout=30,
        )
        r.raise_for_status()
        rows = r.json()
        return rows[0] if rows else None

    def update(self, video_id: str, fields: dict):
        r = requests.patch(
            f"{self._url}/rest/v1/videos?id=eq.{video_id}",
            headers={**self._headers, "Prefer": "return=minimal"},
            json=fields, timeout=30,
        )
        r.raise_for_status()


def progress(db: SupabaseVideos, video_id: str, pct: int):
    db.update(video_id, {"render_progress": pct})


def fail(db: SupabaseVideos, video_id: str, reason: str):
    log.error(f"[{video_id}] FAILED: {reason}")
    db.update(video_id, {"status": "failed", "review_notes": reason})


def download(url: str, dest: str):
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                f.write(chunk)


async def edge_tts_fallback(text: str, dest: str):
    comm = edge_tts.Communicate(text, EDGE_TTS_FALLBACK_VOICE)
    await comm.save(dest)


def run_ffmpeg(*args: str, check=True):
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args]
    log.debug("ffmpeg: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if check and result.returncode != 0:
        raise RuntimeError(f"ffmpeg error: {result.stderr.strip()}")
    return result


def get_media_duration(path: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
        capture_output=True, text=True,
    )
    return float(result.stdout.strip())


def qc_check(path: str, retries: int = 6, retry_delay: float = 5.0) -> tuple[bool, str]:
    """Confirmed live 2026-09-13 on this VPS (1 vCPU): a final.mp4 that
    ffprobe reports as having "no readable video stream" immediately after
    the concat ffmpeg process exits can be a perfectly valid file moments
    later — inspecting the exact same bytes minutes after a "failed" run
    showed a fully valid h264/1080x1920 stream, and a first attempt at
    retrying (3 tries, 2s apart — under 5s total) still failed every time
    despite the file being genuinely fine by the time it was checked
    manually afterward. subprocess.run() only guarantees the child process
    has exited, not that the OS has finished making every write from a
    CPU-starved box fully visible to a sibling process's open() right away,
    and on this box that gap is apparently longer than a few seconds for a
    ~20MB file. Retrying with a real time budget (up to 30s here) before
    declaring a genuine failure is what actually distinguishes "broken
    output" from "this box was just slow to catch up" — the qc_check
    callsite otherwise nukes the whole render (and the ffmpeg work already
    done) over a false alarm.
    """
    last_reason = "unknown"
    for attempt in range(1, retries + 1):
        try:
            dur = get_media_duration(path)
        except Exception as e:
            last_reason = f"could not probe final video duration: {e}"
            dur = None

        if dur is not None:
            if dur < 1.0:
                last_reason = f"final video duration suspiciously short ({dur:.1f}s)"
            else:
                result = subprocess.run(
                    ["ffprobe", "-v", "error", "-select_streams", "v:0",
                     "-show_entries", "stream=width,height,codec_name",
                     "-of", "default=noprint_wrappers=1", path],
                    capture_output=True, text=True,
                )
                # key=value output, not positional CSV — confirmed live this
                # run (2026-09-14): ffprobe's csv=p=0 output for this exact
                # field list came back as "h264,1080,1920" (codec_name FIRST),
                # not the requested "width,height,codec_name" order, so
                # parts[0].isdigit() rejected a perfectly valid 1080x1920
                # video every single retry. Parsing by key instead of
                # position can never be broken by ffprobe's actual field
                # ordering again.
                fields: dict[str, str] = {}
                for line in result.stdout.strip().splitlines():
                    if "=" in line:
                        k, v = line.split("=", 1)
                        fields[k] = v
                width_s, height_s, codec = fields.get("width"), fields.get("height"), fields.get("codec_name")
                if not width_s or not height_s or not codec or not width_s.isdigit() or not height_s.isdigit():
                    last_reason = "final video has no readable video stream"
                    try:
                        st = os.stat(path)
                        size_mtime = f"size={st.st_size}B mtime_age={time.time() - st.st_mtime:.1f}s"
                    except OSError as e:
                        size_mtime = f"stat failed: {e}"
                    log.warning(
                        f"qc_check ffprobe diagnostic: returncode={result.returncode} "
                        f"stdout={result.stdout.strip()!r} stderr={result.stderr.strip()!r} {size_mtime}"
                    )
                else:
                    width, height = int(width_s), int(height_s)
                    if width < 480 or height < 480:
                        last_reason = f"final video resolution too small ({width}x{height})"
                    else:
                        return True, f"{dur:.1f}s, {width}x{height}, {codec}"

        if attempt < retries:
            log.warning(f"qc_check attempt {attempt}/{retries} failed ({last_reason}), retrying in {retry_delay}s")
            time.sleep(retry_delay)

    return False, last_reason


def extend_video_freeze_last_frame(src: str, out: str, pad_seconds: float):
    run_ffmpeg(
        "-i", src,
        "-vf", f"tpad=stop_mode=clone:stop_duration={pad_seconds}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-an",
        out,
    )


def extract_audio(src: str, dest: str) -> bool:
    result = run_ffmpeg("-i", src, "-vn", "-c:a", "aac", dest, check=False)
    return result.returncode == 0 and os.path.exists(dest) and os.path.getsize(dest) > 1000


def silent_audio(duration: float, dest: str):
    run_ffmpeg(
        "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
        "-t", str(duration),
        "-c:a", "aac", "-b:a", "128k",
        dest,
    )


def bg_music_segment(bg_music_path: str, start_offset: float, duration: float, dest: str):
    run_ffmpeg(
        "-stream_loop", "-1", "-i", bg_music_path,
        "-ss", str(start_offset), "-t", str(duration),
        "-c:a", "aac", "-b:a", "128k",
        dest,
    )


def mix_tracks(tracks: list[tuple[str, float]], out_path: str, duration: float):
    if not tracks:
        return False
    inputs: list[str] = []
    filter_parts: list[str] = []
    for i, (path, vol) in enumerate(tracks):
        inputs += ["-i", path]
        filter_parts.append(f"[{i}:a]volume={vol}[a{i}]")
    mix_labels = "".join(f"[a{i}]" for i in range(len(tracks)))
    filter_complex = ";".join(filter_parts) + f";{mix_labels}amix=inputs={len(tracks)}:duration=longest:dropout_transition=2[mixed]"
    run_ffmpeg(
        *inputs,
        "-filter_complex", filter_complex,
        "-map", "[mixed]",
        "-t", str(duration),
        "-c:a", "aac", "-b:a", "128k",
        out_path,
    )
    return True


def image_to_video(image_path: str, audio_path: str | None, duration: float,
                   out_path: str, is_portrait: bool, variant: int = 0):
    w, h = (1080, 1920) if is_portrait else (1920, 1080)
    fps = 24
    frames = max(1, round(duration * fps))
    v = variant % 4
    if v == 0:
        vf = (
            f"scale=2400:-1,"
            f"zoompan=z='min(zoom+0.0015,1.3)':d={frames}:"
            f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={w}x{h}:fps={fps}"
        )
    elif v == 1:
        vf = (
            f"scale=2400:-1,"
            f"zoompan=z='if(eq(on,0),1.3,max(zoom-0.0015,1.0))':d={frames}:"
            f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={w}x{h}:fps={fps}"
        )
    elif v == 2:
        vf = (
            f"scale=2400:-1,"
            f"zoompan=z='1.15':d={frames}:"
            f"x='(iw-iw/zoom)*on/{max(frames-1,1)}':y='ih/2-(ih/zoom/2)':s={w}x{h}:fps={fps}"
        )
    else:
        vf = (
            f"scale=2400:-1,"
            f"zoompan=z='1.15':d={frames}:"
            f"x='(iw-iw/zoom)*(1-on/{max(frames-1,1)})':y='ih/2-(ih/zoom/2)':s={w}x{h}:fps={fps}"
        )
    if audio_path:
        run_ffmpeg(
            "-loop", "1", "-i", image_path,
            "-i", audio_path,
            "-vf", vf,
            "-t", str(duration),
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-c:a", "aac", "-b:a", "128k",
            "-shortest",
            "-pix_fmt", "yuv420p",
            out_path,
        )
    else:
        run_ffmpeg(
            "-loop", "1", "-i", image_path,
            "-t", str(duration),
            "-vf", vf,
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-an",
            "-pix_fmt", "yuv420p",
            out_path,
        )


_CAPTION_FONT = next((
    f for f in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf",
    ) if os.path.exists(f)
), None)


def _wrap_caption(text: str, max_chars: int = 28) -> str:
    words = text.split()
    lines: list[str] = []
    cur = ""
    for w in words:
        candidate = f"{cur} {w}".strip()
        if len(candidate) > max_chars and cur:
            lines.append(cur)
            cur = w
        else:
            cur = candidate
    if cur:
        lines.append(cur)
    return "\n".join(lines)


def finalize_segment(
    src: str, out: str, is_portrait: bool,
    audio_override: str | None = None,
    caption_text: str | None = None,
    captions_style: str = "minimal",
    caption_ass_path: str | None = None,
):
    w, h = (1080, 1920) if is_portrait else (1920, 1080)
    vf_parts = [f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}"]
    vf_parts.append("eq=contrast=1.05:saturation=1.1:gamma=0.98,noise=alls=4:allf=t+u")

    caption_text_path = None
    if caption_ass_path:
        vf_parts.append(f"ass={caption_ass_path}")
    elif caption_text and _CAPTION_FONT:
        caption_text_path = src + ".caption.txt"
        with open(caption_text_path, "w", encoding="utf-8") as f:
            f.write(_wrap_caption(caption_text))

        if captions_style == "highlight":
            box, fontcolor, fontsize = "boxcolor=0xFF9D3D@0.88:boxborderw=10", "0x06060a", 26
        elif captions_style == "bold":
            box, fontcolor, fontsize = "boxcolor=black@0.6:boxborderw=10", "white", 28
        else:  # minimal
            box, fontcolor, fontsize = "boxcolor=black@0.35:boxborderw=6", "white", 22

        vf_parts.append(
            f"drawtext=textfile={caption_text_path}:fontfile={_CAPTION_FONT}:"
            f"fontcolor={fontcolor}:fontsize={fontsize}:{box}:"
            f"x=(w-text_w)/2:y=h-320:line_spacing=4"
        )

    vf_parts.append("drawbox=x=0:y=ih-40:w=120:h=40:color=0x06060a@0.88:t=fill")
    if _CAPTION_FONT:
        vf_parts.append(
            f"drawtext=text='◆ MAXIS':fontfile={_CAPTION_FONT}:"
            f"fontcolor=#FF9D3D:fontsize=12:x=8:y=H-26:shadowcolor=black:shadowx=0:shadowy=1"
        )

    vf = ",".join(vf_parts)
    af = "loudnorm=I=-16:TP=-1.5:LRA=11"

    try:
        if audio_override:
            run_ffmpeg(
                "-i", src, "-i", audio_override,
                "-vf", vf,
                "-r", "24",
                "-map", "0:v", "-map", "1:a",
                "-af", af,
                "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                "-c:a", "aac", "-b:a", "128k",
                "-ar", "44100",
                "-shortest",
                "-pix_fmt", "yuv420p",
                out,
            )
        else:
            run_ffmpeg(
                "-i", src,
                "-vf", vf,
                "-r", "24",
                "-af", af,
                "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                "-c:a", "aac", "-b:a", "128k",
                "-ar", "44100",
                "-pix_fmt", "yuv420p",
                out,
            )
    finally:
        if caption_text_path:
            os.unlink(caption_text_path)


def _concat_hardcut(segment_paths: list[str], out_path: str):
    list_file = out_path + ".list.txt"
    with open(list_file, "w") as f:
        for p in segment_paths:
            f.write(f"file '{p}'\n")
    run_ffmpeg(
        "-f", "concat", "-safe", "0",
        "-i", list_file,
        "-c", "copy",
        out_path,
    )
    os.unlink(list_file)


def _concat_crossfade(segment_paths: list[str], out_path: str, transition: float):
    durations = [get_media_duration(p) for p in segment_paths]

    inputs: list[str] = []
    for p in segment_paths:
        inputs += ["-i", p]

    filter_parts: list[str] = []
    prev_v, prev_a = "0:v", "0:a"
    cumulative = durations[0]
    for i in range(1, len(segment_paths)):
        d = max(0.05, min(transition, durations[i - 1], durations[i]))
        offset = max(0.0, cumulative - d)
        vout, aout = f"v{i}", f"a{i}"
        filter_parts.append(f"[{prev_v}][{i}:v]xfade=transition=fade:duration={d:.3f}:offset={offset:.3f}[{vout}]")
        filter_parts.append(f"[{prev_a}][{i}:a]acrossfade=d={d:.3f}[{aout}]")
        prev_v, prev_a = vout, aout
        cumulative = offset + durations[i]

    filter_complex = ";".join(filter_parts)
    run_ffmpeg(
        *inputs,
        "-filter_complex", filter_complex,
        "-map", f"[{prev_v}]", "-map", f"[{prev_a}]",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "aac", "-b:a", "128k",
        "-pix_fmt", "yuv420p",
        out_path,
    )


def concat_segments(segment_paths: list[str], out_path: str, transition: float = 0.3):
    if len(segment_paths) <= 1:
        _concat_hardcut(segment_paths, out_path)
        return
    try:
        _concat_crossfade(segment_paths, out_path, transition)
    except Exception as e:
        log.warning(f"Crossfade concat failed ({e}), falling back to hard-cut concat")
        _concat_hardcut(segment_paths, out_path)


def save_final_video(local_path: str, video_id: str) -> str:
    """No AWS/S3 — this VPS IS the storage now (see clip_worker.py's
    identical rationale).

    Two execution contexts share this one function:
    - On Browsight itself (BROWSIGHT_UPLOAD_URL unset): write straight into
      local media storage via direct import, as always.
    - On a GitHub Actions runner (stitching moved there 2026-09-14 to get
      real multi-core CPU for the ffmpeg-heavy encode/concat steps, which
      this 1-vCPU VPS was starving on): there is no local media_store to
      write into, so instead call Browsight's existing presigned-upload
      pair (api_storage_presign / api_storage_direct_upload in app.py —
      the same mechanism already used for large client-side video uploads
      from the browser) to PUT the finished file back to this VPS over
      HTTPS. Same bucket_path, same returned URL shape either way.
    """
    bucket_path = f"film/{video_id}/final.mp4"
    upload_url = os.environ.get("BROWSIGHT_UPLOAD_URL")
    api_key = os.environ.get("BROWSIGHT_API_KEY")

    if upload_url and api_key:
        presign = requests.post(
            f"{upload_url.rstrip('/')}/api/storage/presign",
            headers={"X-API-Key": api_key, "Content-Type": "application/json"},
            json={"key": bucket_path, "contentType": "video/mp4"},
            timeout=30,
        )
        presign.raise_for_status()
        put_url = presign.json()["uploadUrl"]
        with open(local_path, "rb") as f:
            put = requests.put(put_url, data=f, timeout=300)
        put.raise_for_status()
        return put.json()["url"]

    sys.path.insert(0, str(ROOT.parent / "dashboard"))
    import storage as media_storage
    with open(local_path, "rb") as f:
        media_storage.save_file(bucket_path, f.read())
    return media_storage.local_url(bucket_path)


# ── Core stitch pipeline (identical logic to server/stitch_worker.py) ──────────

async def stitch_video(db: SupabaseVideos, video: dict):
    video_id = video["id"]
    scenes: list[dict] = video.get("scenes") or []
    is_portrait = (video.get("video_format") or "portrait") == "portrait"
    film_settings = video.get("film_settings") or {}
    bg_music_url = film_settings.get("bg_music_url")
    bg_music_volume = (film_settings.get("bg_music_volume") if film_settings.get("bg_music_volume") is not None else 30) / 100.0
    captions_enabled = bool(film_settings.get("captions_enabled"))
    captions_style = film_settings.get("captions_style") or "minimal"

    approved = [
        s for s in scenes
        if s.get("clip_status") == "approved"
        and (s.get("clip_url") or s.get("reference_image_url"))
    ]
    approved.sort(key=lambda s: s.get("order", 0))

    if not approved:
        fail(db, video_id, "No approved scenes found when worker picked up job")
        return

    log.info(f"[{video_id}] Stitching {len(approved)} scenes (portrait={is_portrait})")
    db.update(video_id, {"status": "stitching", "render_progress": 5})

    # dir="/var/tmp" (real ext4 disk, 32GB free), not the default /tmp: this
    # box's /tmp is a 1.7GB tmpfs mount, meaning every downloaded clip,
    # normalized segment, and the final stitched video would otherwise
    # compete directly against process RAM for the same memory pool on a
    # 1-vCPU/3.3GB box that has already been observed swapping under this
    # exact workload (swap hit 9GB this session). Confirmed via `mount`:
    # tmpfs on /tmp, ext4 on /.
    tmpdir = tempfile.mkdtemp(prefix=f"maxis_{video_id}_", dir="/var/tmp")
    try:
        segments: list[str] = []

        bg_music_path = None
        if bg_music_url:
            bg_music_path = os.path.join(tmpdir, "bg_music_src")
            try:
                download(bg_music_url, bg_music_path)
                log.info(f"[{video_id}] Background music downloaded (volume={bg_music_volume})")
            except Exception as e:
                log.warning(f"[{video_id}] Failed to download bg_music_url: {e}")
                bg_music_path = None

        elapsed = 0.0

        for idx, scene in enumerate(approved):
            scene_id = scene.get("id", f"scene_{idx}")
            clip_url = scene.get("clip_url")
            img_url  = scene.get("reference_image_url")
            dialogue = (scene.get("dialogue") or "").strip()
            narration = (scene.get("narration") or "").strip()
            caption_text = dialogue or narration
            duration = float(scene.get("duration_s") or 8)
            seg_norm = os.path.join(tmpdir, f"seg_{idx:03d}_norm.mp4")

            caption_ass_path = None
            captions_url = scene.get("clip_captions_url")
            if captions_url and captions_enabled:
                caption_ass_path = os.path.join(tmpdir, f"cap_{idx:03d}.ass")
                try:
                    download(captions_url, caption_ass_path)
                except Exception as e:
                    log.warning(f"  [{idx+1}/{len(approved)}] Failed to download clip_captions_url, falling back to static caption: {e}")
                    caption_ass_path = None

            voice_path = None
            audio_url = scene.get("clip_audio_url")
            if audio_url:
                voice_path = os.path.join(tmpdir, f"voice_{idx:03d}")
                try:
                    download(audio_url, voice_path)
                except Exception as e:
                    log.warning(f"  [{idx+1}/{len(approved)}] Failed to download clip_audio_url: {e}")
                    voice_path = None
            if not voice_path and dialogue and not clip_url:
                voice_path = os.path.join(tmpdir, f"voice_{idx:03d}.mp3")
                log.warning(f"  [{idx+1}/{len(approved)}] dialogue present but no usable clip_audio_url — falling back to generic edge-tts voice")
                try:
                    await edge_tts_fallback(dialogue, voice_path)
                except Exception as e:
                    log.warning(f"  [{idx+1}/{len(approved)}] edge-tts fallback also failed: {e}")
                    voice_path = None

            effective_duration = duration
            if voice_path:
                try:
                    effective_duration = max(duration, get_media_duration(voice_path))
                except Exception as e:
                    log.warning(f"  [{idx+1}/{len(approved)}] couldn't measure voice duration, using nominal: {e}")

            bg_seg_path = None
            if bg_music_path:
                bg_seg_path = os.path.join(tmpdir, f"bg_{idx:03d}.aac")
                try:
                    bg_music_segment(bg_music_path, elapsed, effective_duration, bg_seg_path)
                except Exception as e:
                    log.warning(f"  [{idx+1}/{len(approved)}] bg music segment failed: {e}")
                    bg_seg_path = None
            elapsed += effective_duration

            if clip_url:
                raw = os.path.join(tmpdir, f"raw_{idx:03d}.mp4")
                log.info(f"  [{idx+1}/{len(approved)}] Downloading clip {scene_id}")
                download(clip_url, raw)

                native_audio = os.path.join(tmpdir, f"native_{idx:03d}.aac")
                has_native = extract_audio(raw, native_audio)

                try:
                    native_video_len = get_media_duration(raw)
                    pad = effective_duration - native_video_len
                    if pad > 0.05:
                        extended = os.path.join(tmpdir, f"extended_{idx:03d}.mp4")
                        extend_video_freeze_last_frame(raw, extended, pad)
                        raw = extended
                except Exception as e:
                    log.warning(f"  [{idx+1}/{len(approved)}] couldn't extend clip to match voiceover: {e}")

                tracks: list[tuple[str, float]] = []
                if has_native: tracks.append((native_audio, 0.55))
                if voice_path:  tracks.append((voice_path, 1.0))
                if bg_seg_path: tracks.append((bg_seg_path, bg_music_volume))

                mixed = os.path.join(tmpdir, f"mixed_{idx:03d}.aac")
                if tracks:
                    mix_tracks(tracks, mixed, effective_duration)
                else:
                    silent_audio(effective_duration, mixed)
                finalize_segment(
                    raw, seg_norm, is_portrait, audio_override=mixed,
                    caption_text=caption_text if captions_enabled else None,
                    captions_style=captions_style,
                    caption_ass_path=caption_ass_path,
                )

            elif img_url:
                ext = ".jpg" if ".jpg" in img_url.lower() else ".png"
                img_path = os.path.join(tmpdir, f"img_{idx:03d}{ext}")
                log.info(f"  [{idx+1}/{len(approved)}] Downloading image {scene_id}")
                download(img_url, img_path)

                tracks = []
                if voice_path:  tracks.append((voice_path, 1.0))
                if bg_seg_path: tracks.append((bg_seg_path, bg_music_volume))

                final_audio = os.path.join(tmpdir, f"mixed_{idx:03d}.aac")
                if tracks:
                    mix_tracks(tracks, final_audio, effective_duration)
                else:
                    silent_audio(effective_duration, final_audio)

                raw = os.path.join(tmpdir, f"raw_{idx:03d}.mp4")
                image_to_video(img_path, final_audio, effective_duration, raw, is_portrait, variant=idx)
                finalize_segment(
                    raw, seg_norm, is_portrait,
                    caption_text=caption_text if captions_enabled else None,
                    captions_style=captions_style,
                    caption_ass_path=caption_ass_path,
                )

            segments.append(seg_norm)
            pct = 10 + int(70 * (idx + 1) / len(approved))
            progress(db, video_id, pct)

        log.info(f"[{video_id}] Concatenating {len(segments)} segments")
        progress(db, video_id, 82)
        final_path = os.path.join(tmpdir, "final.mp4")
        concat_segments(segments, final_path)

        qc_ok, qc_msg = qc_check(final_path)
        if not qc_ok:
            raise RuntimeError(f"QC failed: {qc_msg}")
        log.info(f"[{video_id}] QC passed: {qc_msg}")

        log.info(f"[{video_id}] Saving final video")
        progress(db, video_id, 92)
        public_url = save_final_video(final_path, video_id)

        db.update(video_id, {
            "status": "render_complete",
            "rendered_video_url": public_url,
            "render_progress": 100,
        })
        log.info(f"[{video_id}] Done → {public_url}")
        print(json.dumps({"success": True, "videoUrl": public_url}), flush=True)

    except Exception as e:
        fail(db, video_id, str(e))
        log.exception(f"[{video_id}] Unexpected error")
        print(json.dumps({"success": False, "error": str(e)}), flush=True)
        sys.exit(1)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video-id", required=True)
    p.add_argument("--supabase-url", required=True)
    p.add_argument("--supabase-key", required=True)
    args = p.parse_args()

    db = SupabaseVideos(args.supabase_url, args.supabase_key)
    video = db.get(args.video_id)
    if not video:
        print(json.dumps({"success": False, "error": "Video not found"}), flush=True)
        sys.exit(1)
    if video.get("status") != "clips_approved":
        print(json.dumps({"success": False, "error": f"Video is not in clips_approved status (currently: {video.get('status')})"}), flush=True)
        sys.exit(1)

    asyncio.run(stitch_video(db, video))


if __name__ == "__main__":
    main()
