import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from .db import add_history, claim_next_file, finish_file, get_setting
from .probe import is_file_stable, probe_file

WORKER_STATE: dict[int, dict[str, Any]] = {}
_STATE_LOCK = threading.Lock()
_TIME_RE = re.compile(r"out_time_us=(\d+)")


def _set_worker(gpu_id: int, state: dict[str, Any]) -> None:
    with _STATE_LOCK:
        WORKER_STATE[gpu_id] = state


def _check_temp_space(temp_dir: Path, min_gb: float) -> bool:
    try:
        temp_dir.mkdir(parents=True, exist_ok=True)
        usage = shutil.disk_usage(temp_dir)
        return (usage.free / (1024**3)) >= min_gb
    except OSError:
        return False


def _needs_transcode(info, target_codec: str, target_container: str, add_aac_stereo: bool) -> tuple[bool, bool, bool]:
    """Returns (needs_work, reencode_video, add_aac)."""
    is_target_video = (
        (target_codec == "hevc" and info.video_codec in ("hevc", "h265") and "10" not in info.pix_fmt)
        or (target_codec == "h264" and info.video_codec == "h264" and "10" not in info.pix_fmt)
    )
    reencode_video = not is_target_video
    add_aac = bool(add_aac_stereo and info.audio_codecs and not info.has_aac_stereo)
    needs_container = info.container != target_container
    return (reencode_video or add_aac or needs_container), reencode_video, add_aac


def _build_ffmpeg_cmd(
    src: Path,
    dst: Path,
    info,
    gpu_id: int,
    target_codec: str,
    reencode_video: bool,
    add_aac: bool,
    cq: str,
    preset: str,
) -> list[str]:
    cmd = ["ffmpeg", "-y", "-hide_banner", "-progress", "pipe:1", "-nostats"]
    if reencode_video and info.video_codec not in ("av1", "vp9", "vp8", "theora"):
        cmd += ["-hwaccel", "cuda", "-hwaccel_device", str(gpu_id)]
    cmd += ["-i", str(src)]

    # Map primary video, all audio, all text/compatible subtitles
    cmd += ["-map", "0:v:0", "-map", "0:a?"]
    if add_aac:
        # Duplicate first audio stream as AAC stereo compatibility track (stream 0)
        cmd += ["-map", "0:a:0"]
    cmd += ["-map", "0:s?"]

    if reencode_video:
        enc = "hevc_nvenc" if target_codec == "hevc" else "h264_nvenc"
        # Pascal P100: 8-bit yuv420p, no B-frames for hevc_nvenc, cap maxrate at 80% of source bitrate
        cmd += ["-c:v", enc, "-gpu", str(gpu_id), "-pix_fmt", "yuv420p", "-rc", "vbr", "-cq", cq, "-b:v", "0", "-preset", preset]
        try:
            src_bps = int((src.stat().st_size * 8) / max(info.duration, 1.0))
            max_kbps = max(600, int((src_bps * 0.80) / 1000))
            cmd += ["-maxrate", f"{max_kbps}k", "-bufsize", f"{max_kbps * 2}k"]
        except Exception:
            pass
        if enc == "hevc_nvenc":
            cmd += ["-bf", "0"]
    else:
        cmd += ["-c:v", "copy"]

    if add_aac:
        # Keep existing audio streams copied, encode the added compatibility track as AAC 2.0
        n_audio = len(info.audio_codecs)
        cmd += ["-c:a", "copy", f"-c:a:{n_audio}", "aac", f"-ac:a:{n_audio}", "2", f"-b:a:{n_audio}", "192k"]
    else:
        cmd += ["-c:a", "copy"]

    cmd += ["-c:s", "copy", "-max_muxing_queue_size", "4096", str(dst)]
    return cmd


def process_one(job: dict[str, Any], gpu_id: int) -> None:
    file_id = job["id"]
    src = Path(job["path"])
    lib_id = job["library_id"]
    target_codec = job.get("target_codec", "hevc")
    target_container = job.get("target_container", "mkv")
    add_aac_stereo = bool(job.get("add_aac_stereo", 1))
    temp_dir = Path(get_setting("temp_dir", "/temp"))
    cq = get_setting("cq", "23")
    preset = get_setting("preset", "p4")

    t0 = time.time()
    _set_worker(gpu_id, {"file": src.name, "path": str(src), "pct": 0.0, "status": "probing", "started": t0})

    if not src.exists():
        finish_file(file_id, "error", error_msg="File missing")
        _set_worker(gpu_id, {"status": "idle"})
        return

    if not is_file_stable(src, wait_sec=2.0):
        # Re-queue if actively changing
        finish_file(file_id, "queued", error_msg="")
        _set_worker(gpu_id, {"status": "idle"})
        return

    try:
        info = probe_file(src)
    except Exception as e:
        finish_file(file_id, "error", error_msg=f"ffprobe failed: {e}")
        add_history(str(src), lib_id, "", "", src.stat().st_size, 0, time.time() - t0, gpu_id, "error", str(e))
        _set_worker(gpu_id, {"status": "idle"})
        return

    needs_work, reencode_video, add_aac = _needs_transcode(info, target_codec, target_container, add_aac_stereo)
    if not needs_work:
        st = src.stat()
        finish_file(
            file_id,
            "skipped",
            size_bytes=st.st_size,
            mtime=st.st_mtime,
            video_codec=info.video_codec,
            audio_codecs=",".join(info.audio_codecs),
            width=info.width,
            height=info.height,
            duration=info.duration,
        )
        _set_worker(gpu_id, {"status": "idle"})
        return

    tmp_out = temp_dir / f"yarr_{file_id}_{int(t0)}.{target_container}"
    final_dst = src.with_suffix(f".{target_container}")
    cmd = _build_ffmpeg_cmd(src, tmp_out, info, gpu_id, target_codec, reencode_video, add_aac, cq, preset)

    _set_worker(gpu_id, {"file": src.name, "path": str(src), "pct": 1.0, "status": "encoding", "started": t0})
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            m = _TIME_RE.search(line)
            if m and info.duration > 0:
                sec = int(m.group(1)) / 1_000_000.0
                pct = min(99.5, round((sec / info.duration) * 100.0, 1))
                _set_worker(gpu_id, {"file": src.name, "path": str(src), "pct": pct, "status": "encoding", "started": t0})
        rc = proc.wait()
        stderr = proc.stderr.read() if proc.stderr else ""
        if rc != 0 or not tmp_out.exists() or tmp_out.stat().st_size < 10240:
            raise RuntimeError(f"ffmpeg rc={rc}: {stderr[-400:]}")

        # Verify output integrity with ffprobe (absorbed from Unmanic postprocessor)
        out_info = probe_file(tmp_out)
        if not out_info.video_codec or (info.duration > 10 and abs(out_info.duration - info.duration) > max(5.0, info.duration * 0.05)):
            raise RuntimeError(f"Output verification failed (dur {out_info.duration:.1f}s vs {info.duration:.1f}s)")

        bytes_before = src.stat().st_size
        bytes_after = tmp_out.stat().st_size
        bytes_saved = bytes_before - bytes_after

        # Size Guard (absorbed from Unmanic/Tdarr): never replace with a larger file
        if reencode_video and bytes_after >= int(bytes_before * 0.98):
            tmp_out.unlink(missing_ok=True)
            st_orig = src.stat()
            finish_file(
                file_id,
                "skipped",
                size_bytes=st_orig.st_size,
                mtime=st_orig.st_mtime,
                video_codec=info.video_codec,
                audio_codecs=",".join(info.audio_codecs),
                width=info.width,
                height=info.height,
                duration=info.duration,
                bytes_saved=0,
            )
            add_history(
                str(src), lib_id, info.video_codec, info.video_codec,
                bytes_before, bytes_before, time.time() - t0, gpu_id, "skipped",
                f"Kept original (encoded {bytes_after/(1024**2):.1f}MB >= source {bytes_before/(1024**2):.1f}MB)",
            )
            _set_worker(gpu_id, {"status": "idle"})
            return

        # Atomic replacement: copy to .yarrtmp in target dir, fsync/rename, then remove old if extension changed
        _set_worker(gpu_id, {"file": src.name, "path": str(src), "pct": 99.9, "status": "replacing", "started": t0})
        stage_dst = final_dst.with_name(f".{final_dst.name}.yarrtmp")
        shutil.copy2(tmp_out, stage_dst)
        os.replace(stage_dst, final_dst)
        if src != final_dst and src.exists():
            src.unlink()
        tmp_out.unlink(missing_ok=True)

        st_new = final_dst.stat()
        finish_file(
            file_id,
            "done",
            path=str(final_dst),
            size_bytes=st_new.st_size,
            mtime=st_new.st_mtime,
            video_codec=out_info.video_codec,
            audio_codecs=",".join(out_info.audio_codecs),
            width=out_info.width,
            height=out_info.height,
            duration=out_info.duration,
            bytes_saved=bytes_saved,
        )
        add_history(
            str(final_dst), lib_id, info.video_codec, out_info.video_codec,
            bytes_before, bytes_after, time.time() - t0, gpu_id, "done",
            f"Saved {bytes_saved / (1024**2):.1f} MB",
        )
    except Exception as e:
        tmp_out.unlink(missing_ok=True)
        finish_file(file_id, "error", error_msg=str(e)[:500])
        add_history(str(src), lib_id, info.video_codec, "", src.stat().st_size if src.exists() else 0, 0, time.time() - t0, gpu_id, "error", str(e)[:300])
    finally:
        _set_worker(gpu_id, {"status": "idle"})


def worker_loop(gpu_id: int, stop_event: threading.Event) -> None:
    _set_worker(gpu_id, {"status": "idle"})
    while not stop_event.is_set():
        if get_setting("paused", "false").lower() == "true":
            _set_worker(gpu_id, {"status": "paused"})
            time.sleep(3)
            continue
        temp_dir = Path(get_setting("temp_dir", "/temp"))
        min_gb = float(get_setting("min_free_temp_gb", "10"))
        if not _check_temp_space(temp_dir, min_gb):
            _set_worker(gpu_id, {"status": "disk_low"})
            time.sleep(10)
            continue
        job = claim_next_file(gpu_id)
        if not job:
            _set_worker(gpu_id, {"status": "idle"})
            time.sleep(3)
            continue
        process_one(job, gpu_id)
