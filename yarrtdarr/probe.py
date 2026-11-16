import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class MediaInfo:
    video_codec: str
    pix_fmt: str
    width: int
    height: int
    duration: float
    audio_codecs: list[str]
    audio_channels: list[int]
    has_aac_stereo: bool
    subtitle_codecs: list[str]
    container: str
    raw: dict[str, Any]


def is_file_stable(path: Path, wait_sec: float = 3.0) -> bool:
    """Absorbed from Unmanic eventmonitor/filetest: ensure file isn't actively being written."""
    try:
        st1 = path.stat()
        if st1.st_size < 1024 * 1024:  # < 1MB skip/ignore tiny stubs
            return False
        time.sleep(wait_sec)
        st2 = path.stat()
        return st1.st_size == st2.st_size and st1.st_mtime == st2.st_mtime
    except OSError:
        return False


def probe_file(path: Path) -> MediaInfo:
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    out = subprocess.check_output(cmd, timeout=45)
    data = json.loads(out)
    streams = data.get("streams", [])
    fmt = data.get("format", {})

    vcodec = ""
    pix_fmt = ""
    width = 0
    height = 0
    acodecs: list[str] = []
    achannels: list[int] = []
    has_aac_stereo = False
    scodecs: list[str] = []

    for s in streams:
        st = s.get("codec_type")
        if st == "video" and not vcodec and s.get("codec_name") not in ("mjpeg", "png", "webp"):
            vcodec = (s.get("codec_name") or "").lower()
            pix_fmt = (s.get("pix_fmt") or "").lower()
            width = int(s.get("width") or 0)
            height = int(s.get("height") or 0)
        elif st == "audio":
            ac = (s.get("codec_name") or "").lower()
            ch = int(s.get("channels") or 0)
            acodecs.append(ac)
            achannels.append(ch)
            if ac == "aac" and ch <= 2:
                has_aac_stereo = True
        elif st == "subtitle":
            scodecs.append((s.get("codec_name") or "").lower())

    duration = float(fmt.get("duration") or 0.0)
    container = path.suffix.lstrip(".").lower()
    return MediaInfo(
        video_codec=vcodec,
        pix_fmt=pix_fmt,
        width=width,
        height=height,
        duration=duration,
        audio_codecs=acodecs,
        audio_channels=achannels,
        has_aac_stereo=has_aac_stereo,
        subtitle_codecs=scodecs,
        container=container,
        raw=data,
    )
