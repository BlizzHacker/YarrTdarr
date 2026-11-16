import os
import threading
import time
from pathlib import Path

from .db import get_conn, queue_file

VIDEO_EXTS = {".mkv", ".mp4", ".mov", ".m4v", ".avi", ".wmv", ".flv", ".ts", ".m2ts", ".webm", ".mpg", ".mpeg"}
IGNORE_PARTS = (".yarrtmp", ".part", ".!qb", ".tmp", "sample")


def scan_library(lib_id: str, folder: str) -> int:
    root = Path(folder)
    if not root.exists():
        return 0
    added = 0
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            low = fn.lower()
            if any(p in low for p in IGNORE_PARTS):
                continue
            ext = Path(low).suffix
            if ext not in VIDEO_EXTS:
                continue
            p = Path(dirpath) / fn
            try:
                st = p.stat()
                if st.st_size < 1024 * 1024:
                    continue
                if queue_file(lib_id, str(p), st.st_size, st.st_mtime):
                    added += 1
            except OSError:
                continue
    conn = get_conn()
    conn.execute("UPDATE libraries SET last_scan_at=? WHERE id=?", (time.time(), lib_id))
    conn.commit()
    conn.close()
    return added


def scanner_loop(stop_event: threading.Event) -> None:
    while not stop_event.is_set():
        conn = get_conn()
        libs = conn.execute("SELECT id, folder, scan_interval_sec, last_scan_at FROM libraries WHERE enabled=1").fetchall()
        conn.close()
        now = time.time()
        for lib in libs:
            if stop_event.is_set():
                break
            if now - (lib["last_scan_at"] or 0) >= (lib["scan_interval_sec"] or 300):
                scan_library(lib["id"], lib["folder"])
        time.sleep(15)
