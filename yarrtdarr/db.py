import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

DB_PATH = Path(os.getenv("YARRTDARR_DB", "/data/yarrtdarr.db"))
_LOCK = threading.Lock()


def get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def _ensure_column(conn: sqlite3.Connection, table: str, col: str, ddl: str) -> None:
    cols = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if col not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {ddl}")


def init_db() -> None:
    with _LOCK:
        conn = get_conn()
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS libraries (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                folder TEXT NOT NULL UNIQUE,
                enabled INTEGER NOT NULL DEFAULT 1,
                target_codec TEXT NOT NULL DEFAULT 'hevc',
                target_container TEXT NOT NULL DEFAULT 'mkv',
                add_aac_stereo INTEGER NOT NULL DEFAULT 1,
                scan_interval_sec INTEGER NOT NULL DEFAULT 300,
                last_scan_at REAL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                library_id TEXT NOT NULL,
                path TEXT NOT NULL UNIQUE,
                size_bytes INTEGER DEFAULT 0,
                mtime REAL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'queued',
                priority INTEGER NOT NULL DEFAULT 0,
                video_codec TEXT DEFAULT '',
                audio_codecs TEXT DEFAULT '',
                width INTEGER DEFAULT 0,
                height INTEGER DEFAULT 0,
                duration REAL DEFAULT 0,
                error_msg TEXT DEFAULT '',
                gpu_id INTEGER DEFAULT -1,
                bytes_saved INTEGER DEFAULT 0,
                updated_at REAL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_files_status ON files(status);
            CREATE INDEX IF NOT EXISTS idx_files_lib ON files(library_id);
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT NOT NULL,
                library_id TEXT NOT NULL,
                codec_before TEXT,
                codec_after TEXT,
                bytes_before INTEGER,
                bytes_after INTEGER,
                bytes_saved INTEGER,
                duration_sec REAL,
                gpu_id INTEGER,
                status TEXT,
                message TEXT,
                ffmpeg_cmd TEXT DEFAULT '',
                finished_at REAL
            );
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )
        _ensure_column(conn, "files", "priority", "INTEGER NOT NULL DEFAULT 0")
        _ensure_column(conn, "history", "ffmpeg_cmd", "TEXT DEFAULT ''")
        defaults = {
            "paused": "false",
            "gpu_workers": os.getenv("YARRTDARR_GPU_WORKERS", "4"),
            "temp_dir": os.getenv("YARRTDARR_TEMP_DIR", "/mnt/lvm_shared/yarrtdarr-temp"),
            "cq": os.getenv("YARRTDARR_CQ", "27"),
            "preset": os.getenv("YARRTDARR_PRESET", "p4"),
            "maxrate_pct": os.getenv("YARRTDARR_MAXRATE_PCT", "80"),
            "min_free_temp_gb": os.getenv("YARRTDARR_MIN_FREE_GB", "10"),
            "size_guard": "true",
            "auto_accept": "true",
            "plex_url": "",
            "plex_token": "",
            "discord_webhook": "",
        }
        for k, v in defaults.items():
            conn.execute("INSERT OR IGNORE INTO settings(key, value) VALUES(?, ?)", (k, v))
        conn.commit()
        conn.close()


def get_setting(key: str, default: str = "") -> str:
    with _LOCK:
        conn = get_conn()
        row = conn.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        conn.close()
        return row["value"] if row else default


def get_all_settings() -> dict[str, str]:
    with _LOCK:
        conn = get_conn()
        rows = conn.execute("SELECT key, value FROM settings").fetchall()
        conn.close()
        return {r["key"]: r["value"] for r in rows}


def set_setting(key: str, value: str) -> None:
    with _LOCK:
        conn = get_conn()
        conn.execute("INSERT OR REPLACE INTO settings(key, value) VALUES(?, ?)", (key, str(value)))
        conn.commit()
        conn.close()


def upsert_library(
    lib_id: str,
    name: str,
    folder: str,
    target_codec: str = "hevc",
    target_container: str = "mkv",
    add_aac_stereo: int = 1,
    scan_interval_sec: int = 300,
    enabled: int = 1,
) -> None:
    with _LOCK:
        conn = get_conn()
        conn.execute(
            """
            INSERT INTO libraries(id, name, folder, enabled, target_codec, target_container, add_aac_stereo, scan_interval_sec)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name=excluded.name,
                folder=excluded.folder,
                enabled=excluded.enabled,
                target_codec=excluded.target_codec,
                target_container=excluded.target_container,
                add_aac_stereo=excluded.add_aac_stereo,
                scan_interval_sec=excluded.scan_interval_sec
            """,
            (lib_id, name, folder, enabled, target_codec, target_container, add_aac_stereo, scan_interval_sec),
        )
        conn.commit()
        conn.close()


def queue_file(library_id: str, path: str, size_bytes: int, mtime: float) -> bool:
    with _LOCK:
        conn = get_conn()
        existing = conn.execute("SELECT id, size_bytes, mtime, status FROM files WHERE path=?", (path,)).fetchone()
        if existing:
            if existing["size_bytes"] == size_bytes and abs(existing["mtime"] - mtime) < 1.0:
                conn.close()
                return False
            conn.execute(
                "UPDATE files SET size_bytes=?, mtime=?, status='queued', error_msg='', updated_at=? WHERE id=?",
                (size_bytes, mtime, time.time(), existing["id"]),
            )
            conn.commit()
            conn.close()
            return True
        conn.execute(
            "INSERT INTO files(library_id, path, size_bytes, mtime, status, updated_at) VALUES(?,?,?,?,'queued',?)",
            (library_id, path, size_bytes, mtime, time.time()),
        )
        conn.commit()
        conn.close()
        return True


def claim_next_file(gpu_id: int) -> dict[str, Any] | None:
    with _LOCK:
        conn = get_conn()
        row = conn.execute(
            """
            SELECT f.*, l.target_codec, l.target_container, l.add_aac_stereo
            FROM files f JOIN libraries l ON f.library_id = l.id
            WHERE f.status = 'queued' AND l.enabled = 1
            ORDER BY f.priority DESC, f.id ASC LIMIT 1
            """
        ).fetchone()
        if not row:
            conn.close()
            return None
        conn.execute(
            "UPDATE files SET status='processing', gpu_id=?, updated_at=? WHERE id=?",
            (gpu_id, time.time(), row["id"]),
        )
        conn.commit()
        d = dict(row)
        conn.close()
        return d


def finish_file(
    file_id: int,
    status: str,
    *,
    path: str | None = None,
    size_bytes: int | None = None,
    mtime: float | None = None,
    video_codec: str = "",
    audio_codecs: str = "",
    width: int = 0,
    height: int = 0,
    duration: float = 0.0,
    bytes_saved: int = 0,
    error_msg: str = "",
) -> None:
    with _LOCK:
        conn = get_conn()
        cur = conn.execute("SELECT * FROM files WHERE id=?", (file_id,)).fetchone()
        if cur:
            conn.execute(
                """
                UPDATE files SET
                    path=COALESCE(?, path),
                    size_bytes=COALESCE(?, size_bytes),
                    mtime=COALESCE(?, mtime),
                    status=?,
                    priority=0,
                    video_codec=?,
                    audio_codecs=?,
                    width=?,
                    height=?,
                    duration=?,
                    bytes_saved=?,
                    error_msg=?,
                    updated_at=?
                WHERE id=?
                """,
                (
                    path, size_bytes, mtime, status, video_codec, audio_codecs,
                    width, height, duration, bytes_saved, error_msg, time.time(), file_id,
                ),
            )
            conn.commit()
        conn.close()


def add_history(
    path: str,
    library_id: str,
    codec_before: str,
    codec_after: str,
    bytes_before: int,
    bytes_after: int,
    duration_sec: float,
    gpu_id: int,
    status: str,
    message: str = "",
    ffmpeg_cmd: str = "",
) -> None:
    with _LOCK:
        conn = get_conn()
        conn.execute(
            """
            INSERT INTO history(path, library_id, codec_before, codec_after, bytes_before, bytes_after, bytes_saved, duration_sec, gpu_id, status, message, ffmpeg_cmd, finished_at)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                path, library_id, codec_before, codec_after, bytes_before, bytes_after,
                bytes_before - bytes_after, duration_sec, gpu_id, status, message, ffmpeg_cmd, time.time(),
            ),
        )
        conn.commit()
        conn.close()
