import os
import subprocess
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .db import get_conn, get_setting, init_db, set_setting, upsert_library
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
    # Reset any interrupted 'processing' rows from prior crash back to 'queued'
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


app = FastAPI(title="YarrTdarr", version="1.0.0", lifespan=lifespan)
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class LibraryIn(BaseModel):
    id: str
    name: str
    folder: str
    target_codec: str = "hevc"


@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/stats")
def api_stats():
    conn = get_conn()
    counts = dict(conn.execute("SELECT status, COUNT(*) FROM files GROUP BY status").fetchall())
    saved = conn.execute("SELECT COALESCE(SUM(bytes_saved), 0) FROM files WHERE status='done'").fetchone()[0]
    libs = [dict(r) for r in conn.execute("SELECT * FROM libraries ORDER BY name").fetchall()]
    conn.close()

    gpus = []
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,utilization.gpu,utilization.encoder,memory.used,memory.total", "--format=csv,noheader,nounits"],
            timeout=5, text=True,
        )
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 6:
                gpus.append({
                    "index": int(parts[0]), "name": parts[1],
                    "gpu_util": int(parts[2]), "enc_util": int(parts[3]),
                    "mem_used": int(parts[4]), "mem_total": int(parts[5]),
                })
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
        "bytes_saved": saved,
        "workers": WORKER_STATE,
        "gpus": gpus,
        "libraries": libs,
    }


@app.get("/api/history")
def api_history(limit: int = 50):
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
    upsert_library(lib.id, lib.name, lib.folder, lib.target_codec)
    return {"ok": True}
