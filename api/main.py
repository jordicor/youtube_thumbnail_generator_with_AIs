"""
FastAPI Main Application

YouTube Thumbnail Generator Web Interface
"""

from fastapi import FastAPI, Request, Query
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager
from pathlib import Path
import sys

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from api.routes import videos, analysis, generation, thumbnails, events, directories, titles
from api.middleware.ip_filter import LANOnlyMiddleware
from database.db import init_db, close_db
from config import MAX_REFERENCE_FRAMES, SERVER_HOST, SERVER_PORT, DEBUG_MODE
from job_queue.client import RedisManager
from job_queue.queue import close_arq_pool


# ============================================================================
# LIFESPAN (startup/shutdown)
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Startup
    await init_db()

    # Check Redis connection
    if await RedisManager.health_check():
        print("Redis connection: OK")
    else:
        print("WARNING: Redis not available - job queue will not work")

    yield

    # Shutdown
    await close_arq_pool()
    await RedisManager.close()
    await close_db()


# ============================================================================
# APP CREATION
# ============================================================================

app = FastAPI(
    title="YouTube Thumbnail Generator",
    description="Generate AI-powered thumbnails with face consistency",
    version="2.0.0",
    lifespan=lifespan
)

# LAN-only IP filtering (must be added first to run before other middleware)
app.add_middleware(LANOnlyMiddleware)

# CORS - Restricted to localhost and common LAN access patterns
ALLOWED_ORIGINS = [
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost:8057",
    "http://127.0.0.1:8057",
    # LAN access - browsers send Origin header with the IP they're connecting to
    # These patterns cover common home network ranges
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=r"^https?://(192\.168\.\d{1,3}\.\d{1,3}|10\.\d{1,3}\.\d{1,3}\.\d{1,3}|172\.(1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})(:\d+)?$",
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
)

# Static files
STATIC_DIR = ROOT_DIR / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Also serve output directory for thumbnails/frames
OUTPUT_DIR = ROOT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output")

# Templates
TEMPLATES_DIR = ROOT_DIR / "templates"
TEMPLATES_DIR.mkdir(exist_ok=True)
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# ============================================================================
# ROUTES
# ============================================================================

# API routes
app.include_router(directories.router, prefix="/api/directories", tags=["directories"])
app.include_router(videos.router, prefix="/api/videos", tags=["videos"])
app.include_router(analysis.router, prefix="/api/analysis", tags=["analysis"])
app.include_router(generation.router, prefix="/api/generation", tags=["generation"])
app.include_router(thumbnails.router, prefix="/api/thumbnails", tags=["thumbnails"])
app.include_router(events.router, prefix="/api/events", tags=["events"])
app.include_router(titles.router, prefix="/api/titles", tags=["titles"])


# ============================================================================
# FRONTEND ROUTES
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Main page - video list."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/video/{video_id}", response_class=HTMLResponse)
async def video_detail(request: Request, video_id: int):
    """Video detail page - clusters and generation."""
    return templates.TemplateResponse(
        "video_detail.html",
        {"request": request, "video_id": video_id}
    )


@app.get("/video/{video_id}/cluster/{cluster_index}/frames", response_class=HTMLResponse)
async def cluster_frames(
    request: Request,
    video_id: int,
    cluster_index: int,
    view_mode: str = Query(default="person", pattern="^(person|person_scene)$")
):
    """Cluster frames management page."""
    return templates.TemplateResponse(
        "cluster_frames.html",
        {
            "request": request,
            "video_id": video_id,
            "cluster_index": cluster_index,
            "max_refs": MAX_REFERENCE_FRAMES,
            "view_mode": view_mode
        }
    )


@app.get("/video/{video_id}/cluster/create", response_class=HTMLResponse)
async def create_cluster(request: Request, video_id: int):
    """Create new cluster manually page."""
    return templates.TemplateResponse(
        "create_cluster.html",
        {"request": request, "video_id": video_id, "max_refs": MAX_REFERENCE_FRAMES}
    )


@app.get("/results/{job_id}", response_class=HTMLResponse)
async def results(request: Request, job_id: int):
    """Results page - generated thumbnails."""
    return templates.TemplateResponse(
        "results.html",
        {"request": request, "job_id": job_id}
    )


# ============================================================================
# HEALTH CHECK
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "version": "2.0.0"}


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=SERVER_HOST,
        port=SERVER_PORT,
        reload=DEBUG_MODE,
        reload_dirs=[str(ROOT_DIR)]
    )
