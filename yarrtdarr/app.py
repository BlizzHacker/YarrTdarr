import shutil
import subprocess
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .db import get_all_settings, get_conn, get_setting, init_db, set_setting, upsert_library
from .transcoder import WORKER_STATE, worker_loop
from .watcher import scan_library, scanner_loop

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
_STOP_EVENT = threading.Event()
_THREADS: list[threading.Thread] = []


def _seed_default_libraries() -> None:
    defaults = [
        ("movies_shared", "Movies (lvm_shared)", "/mnt/lvm_shared/Movie"),
        ("tv_shared", "TV Shows (lvm_shared)", "/mnt/lvm_shared/TV_Show"),
        ("movies_usb2", "Movies (usb2)", "/mnt/usb2/Movie"),
        ("movies4k_usb2", "Movies 4K (usb2)", "/mnt/usb2/Movie4K"),
        ("tv_usb2", "TV Shows (usb2)", "/mnt/usb2/TV_Shows"),
        ("anime_usb2", "Anime (usb2)", "/mnt/usb2/Anime"),
        ("cartoon_usb2", "Cartoon (usb2)", "/mnt/usb2/Cartoon"),
        ("docs_usb2", "Documentaries (usb2)", "/mnt/usb2/Documentaries"),
        ("movies_usb4", "Movies (usb4)", "/mnt/usb4/Movie"),
        ("tv_usb4", "TV Shows (usb4)", "/mnt/usb4/TV_Show"),
        ("tv_legacy_usb4", "TV Legacy (usb4)", "/mnt/usb4/TV"),
        ("movies_usb1", "Movies (usb1)", "/mnt/usb1/Movie"),
        ("tv_usb1", "TV Shows (usb1)", "/mnt/usb1/TV_Show"),
        ("movies_usb3", "Movies (usb3)", "/mnt/usb3/Movie"),
        ("tv_usb3", "TV Shows (usb3)", "/mnt/usb3/TV_Show"),
    ]
    conn = get_conn()
    count = conn.execute("SELECT COUNT(*) FROM libraries").fetchone()[0]
    conn.close()
    if count == 0:
        for lid, name, folder in defaults:
            upsert_library(lid, name, folder, "hevc")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    _seed_default_libraries()
    conn = get_conn()
    conn.execute("UPDATE files SET status='queued' WHERE status='processing'")
    conn.commit()
    conn.close()

    n_gpus = int(get_setting("gpu_workers", "4"))
    for gid in range(n_gpus):
        t = threading.Thread(target=worker_loop, args=(gid, _STOP_EVENT), daemon=True, name=f"gpu-{gid}")
        t.start()
        _THREADS.append(t)
    st = threading.Thread(target=scanner_loop, args=(_STOP_EVENT,), daemon=True, name="scanner")
    st.start()
    _THREADS.append(st)
    yield
    _STOP_EVENT.set()


app = FastAPI(title="YarrTdarr", version="2.0.0", lifespan=lifespan)
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class LibraryIn(BaseModel):
    id: str
    name: str
    folder: str
    target_codec: str = "hevc"
    target_container: str = "mkv"
    add_aac_stereo: int = 1
    scan_interval_sec: int = 300
    enabled: int = 1


@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/stats")
def api_stats():
    conn = get_conn()
    counts = dict(conn.execute("SELECT status, COUNT(*) FROM files GROUP BY status").fetchall())
    saved = conn.execute("SELECT COALESCE(SUM(bytes_saved), 0) FROM files WHERE status='done' AND bytes_saved > 0").fetchone()[0]
    codecs = dict(conn.execute("SELECT COALESCE(NULLIF(video_codec,''), 'pending'), COUNT(*) FROM files GROUP BY 1").fetchall())
    libs = [dict(r) for r in conn.execute("SELECT l.*, (SELECT COUNT(*) FROM files f WHERE f.library_id=l.id) as file_count, (SELECT COUNT(*) FROM files f WHERE f.library_id=l.id AND f.status='done') as done_count, (SELECT COUNT(*) FROM files f WHERE f.library_id=l.id AND f.status='queued') as queued_count FROM libraries l ORDER BY l.name").fetchall()]
    conn.close()

    gpus = []
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,utilization.gpu,utilization.encoder,memory.used,memory.total,temperature.gpu,power.draw", "--format=csv,noheader,nounits"],
            timeout=5, text=True,
        )
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 8:
                gpus.append({
                    "index": int(parts[0]), "name": parts[1],
                    "gpu_util": int(parts[2]), "enc_util": int(parts[3]),
                    "mem_used": int(parts[4]), "mem_total": int(parts[5]),
                    "temp": int(float(parts[6])), "power_w": float(parts[7]),
                })
    except Exception:
        pass

    temp_dir = Path(get_setting("temp_dir", "/mnt/lvm_shared/yarrtdarr-temp"))
    disk_free_gb = 0.0
    try:
        disk_free_gb = round(shutil.disk_usage(temp_dir).free / (1024**3), 1)
    except Exception:
        pass

    return {
        "paused": get_setting("paused", "false") == "true",
        "counts": {
            "queued": counts.get("queued", 0),
            "processing": counts.get("processing", 0),
            "done": counts.get("done", 0),
            "skipped": counts.get("skipped", 0),
            "error": counts.get("error", 0),
            "total": sum(counts.values()),
        },
        "codecs": codecs,
        "bytes_saved": saved,
        "temp_free_gb": disk_free_gb,
        "workers": WORKER_STATE,
        "gpus": gpus,
        "libraries": libs,
    }


@app.get("/api/files")
def api_files(status: str = "all", q: str = "", lib_id: str = "", limit: int = 100):
    conn = get_conn()
    where, params = [], []
    if status != "all":
        where.append("status=?")
        params.append(status)
    if lib_id:
        where.append("library_id=?")
        params.append(lib_id)
    if q:
        where.append("path LIKE ?")
        params.append(f"%{q}%")
    clause = ("WHERE " + " AND ".join(where)) if where else ""
    sql = f"SELECT * FROM files {clause} ORDER BY priority DESC, updated_at DESC, id ASC LIMIT ?"
    params.append(limit)
    rows = [dict(r) for r in conn.execute(sql, params).fetchall()]
    conn.close()
    return rows


@app.post("/api/files/{file_id}/bump")
def api_bump_file(file_id: int):
    conn = get_conn()
    conn.execute("UPDATE files SET priority=100, status='queued', error_msg='' WHERE id=?", (file_id,))
    conn.commit()
    conn.close()
    return {"bumped": file_id}


@app.post("/api/files/{file_id}/requeue")
def api_requeue_file(file_id: int):
    conn = get_conn()
    conn.execute("UPDATE files SET status='queued', error_msg='' WHERE id=?", (file_id,))
    conn.commit()
    conn.close()
    return {"requeued": file_id}


@app.get("/api/history")
def api_history(limit: int = 100):
    conn = get_conn()
    rows = [dict(r) for r in conn.execute("SELECT * FROM history ORDER BY id DESC LIMIT ?", (limit,)).fetchall()]
    conn.close()
    return rows


@app.get("/api/errors")
def api_errors(limit: int = 100):
    conn = get_conn()
    rows = [dict(r) for r in conn.execute("SELECT id, library_id, path, error_msg, updated_at FROM files WHERE status='error' ORDER BY updated_at DESC LIMIT ?", (limit,)).fetchall()]
    conn.close()
    return rows


@app.post("/api/pause")
def api_pause(paused: bool):
    set_setting("paused", "true" if paused else "false")
    return {"paused": paused}


@app.post("/api/scan/{lib_id}")
def api_scan_one(lib_id: str):
    conn = get_conn()
    row = conn.execute("SELECT id, folder FROM libraries WHERE id=?", (lib_id,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(404, "Library not found")
    threading.Thread(target=scan_library, args=(row["id"], row["folder"]), daemon=True).start()
    return {"scanning": lib_id}


@app.post("/api/scan-all")
def api_scan_all():
    conn = get_conn()
    libs = conn.execute("SELECT id, folder FROM libraries WHERE enabled=1").fetchall()
    conn.close()
    for l in libs:
        threading.Thread(target=scan_library, args=(l["id"], l["folder"]), daemon=True).start()
    return {"scanning": len(libs)}


@app.post("/api/retry-errors")
def api_retry_errors():
    conn = get_conn()
    cur = conn.execute("UPDATE files SET status='queued', error_msg='' WHERE status='error'")
    n = cur.rowcount
    conn.commit()
    conn.close()
    return {"requeued": n}


@app.post("/api/libraries")
def api_add_library(lib: LibraryIn):
    upsert_library(lib.id, lib.name, lib.folder, lib.target_codec, lib.target_container, lib.add_aac_stereo, lib.scan_interval_sec, lib.enabled)
    return {"ok": True}


@app.post("/api/libraries/{lib_id}/toggle")
def api_toggle_library(lib_id: str):
    conn = get_conn()
    conn.execute("UPDATE libraries SET enabled = CASE WHEN enabled=1 THEN 0 ELSE 1 END WHERE id=?", (lib_id,))
    conn.commit()
    conn.close()
    return {"ok": True}


@app.delete("/api/libraries/{lib_id}")
def api_delete_library(lib_id: str):
    conn = get_conn()
    conn.execute("DELETE FROM libraries WHERE id=?", (lib_id,))
    conn.commit()
    conn.close()
    return {"deleted": lib_id}


@app.get("/api/settings")
def api_get_settings():
    return get_all_settings()


@app.put("/api/settings")
def api_put_settings(payload: dict[str, Any]):
    for k, v in payload.items():
        set_setting(str(k), str(v))
    return {"ok": True}


@app.get("/api/logs")
def api_logs(lines: int = 80):
    try:
        out = subprocess.check_output(["journalctl", "-u", "yarrtdarr.service", "-n", str(lines), "--no-pager"], timeout=5, text=True)
        return {"lines": out.splitlines()}
    except Exception as e:
        return {"lines": [str(e)]}
