"""
FastAPI Main Application

YouTube Thumbnail Generator Web Interface
"""

from fastapi import FastAPI, Request, Query
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.routing import get_route_path
from contextlib import asynccontextmanager
from pathlib import Path
from pydantic import BaseModel
import sys

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from api.routes import videos, analysis, generation, thumbnails, events, directories, titles, tasks, people, settings
from api.middleware.ip_filter import LANOnlyMiddleware
from api.middleware.request_guard import RequestGuardMiddleware
from database.db import init_db, close_db
from config import (
    DEFAULT_FACE_REFERENCE_COUNT, SERVER_HOST, SERVER_PORT, DEBUG_MODE, CORS_ALLOWED_ORIGINS, CORS_ORIGIN_REGEX,
    OUTPUT_DIR,
)
from services.face_swap_service import face_restoration_available, face_swap_available
from job_queue.client import RedisManager
from job_queue.queue import close_arq_pool
from i18n.i18n import translate as t, set_language, get_language, get_all_translations, SUPPORTED_LANGUAGES, DEFAULT_LANGUAGE


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
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS,
    allow_origin_regex=CORS_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
)

# Only this machine's names, and no cross-site requests that change data
app.add_middleware(RequestGuardMiddleware)


# Static files
class ContainedStaticFiles(StaticFiles):
    r"""Static files that never look outside their folder.

    On Windows a URL path with a backslash or a colon ("\\host\share\file",
    "C:/...") joins into an absolute path, and StaticFiles looks the file up
    before seeing it is outside its folder: looking up a UNC path connects to
    that host and hands it the Windows credentials. No file served here has
    either character, so those paths are not found before any lookup.
    """

    def get_path(self, scope):
        route_path = get_route_path(scope)
        if '\\' in route_path or ':' in route_path:
            raise StarletteHTTPException(status_code=404)
        return super().get_path(scope)


class RevalidatedStaticFiles(ContainedStaticFiles):
    """Static files the browser revalidates on every use (a cheap 304 when
    unchanged), so an updated script or stylesheet is never used stale."""

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers['Cache-Control'] = 'no-cache'
        return response


STATIC_DIR = ROOT_DIR / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", RevalidatedStaticFiles(directory=str(STATIC_DIR)), name="static")

# Also serve output directory for thumbnails/frames
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/output", ContainedStaticFiles(directory=str(OUTPUT_DIR)), name="output")

# Templates
TEMPLATES_DIR = ROOT_DIR / "templates"
TEMPLATES_DIR.mkdir(exist_ok=True)
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Add i18n globals to Jinja2
templates.env.globals["t"] = t
templates.env.globals["get_language"] = get_language
templates.env.globals["get_all_translations"] = get_all_translations
templates.env.globals["SUPPORTED_LANGUAGES"] = SUPPORTED_LANGUAGES


# ============================================================================
# I18N MIDDLEWARE
# ============================================================================

@app.middleware("http")
async def i18n_middleware(request: Request, call_next):
    """Detect language from cookie and set it for the request.

    Thread-safety note: i18n.set_language uses a ContextVar, so concurrent
    requests each get their own language value without racing.
    """
    lang_cookie = request.cookies.get("lang")
    if lang_cookie and lang_cookie in SUPPORTED_LANGUAGES:
        set_language(lang_cookie)
    else:
        set_language(DEFAULT_LANGUAGE)

    response = await call_next(request)
    return response


@app.exception_handler(OverflowError)
async def number_too_large_handler(request: Request, exc: OverflowError):
    """An id beyond SQLite's 64-bit integers names nothing: a client error, not a crash."""
    if 'too large to convert to SQLite INTEGER' not in str(exc):
        raise exc
    return JSONResponse(status_code=422, content={"detail": t('api.errors.number_too_large')})


# ============================================================================
# I18N ENDPOINT
# ============================================================================

class LanguageRequest(BaseModel):
    lang: str


@app.post("/api/lang")
async def set_language_endpoint(request: Request, body: LanguageRequest):
    """Set language preference."""
    if body.lang not in SUPPORTED_LANGUAGES:
        return JSONResponse(
            status_code=400,
            content={"error": f"Unsupported language: {body.lang}"}
        )

    response = JSONResponse(content={"lang": body.lang, "success": True})
    response.set_cookie(
        key="lang",
        value=body.lang,
        max_age=365 * 24 * 60 * 60,  # 1 year
        httponly=False,
        samesite="lax"
    )
    return response


@app.get("/api/lang")
async def get_language_endpoint(request: Request):
    """Get current language."""
    return {"lang": get_language()}


# ============================================================================
# ROUTES
# ============================================================================

# API routes
app.include_router(directories.router, prefix="/api/directories", tags=["directories"])
app.include_router(videos.router, prefix="/api/videos", tags=["videos"])
app.include_router(analysis.router, prefix="/api/analysis", tags=["analysis"])
app.include_router(people.router, prefix="/api/people", tags=["people"])
app.include_router(generation.router, prefix="/api/generation", tags=["generation"])
app.include_router(thumbnails.router, prefix="/api/thumbnails", tags=["thumbnails"])
app.include_router(events.router, prefix="/api/events", tags=["events"])
app.include_router(titles.router, prefix="/api/titles", tags=["titles"])
app.include_router(tasks.router, prefix="/api/tasks", tags=["tasks"])
app.include_router(settings.router, prefix="/api/settings", tags=["settings"])


# ============================================================================
# FRONTEND ROUTES
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Main page - video list."""
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/video/{video_id}", response_class=HTMLResponse)
async def video_detail(request: Request, video_id: int):
    """Video detail page - clusters and generation."""
    # Face swap controls are turned off, saying why, while its models are missing
    _, face_swap_missing = face_swap_available()
    _, face_restoration_missing = face_restoration_available()
    return templates.TemplateResponse(
        request=request, name="video_detail.html",
        context={"video_id": video_id, "face_swap_missing": face_swap_missing,
         "face_restoration_missing": face_restoration_missing}
    )


@app.get("/video/{video_id}/person/new", response_class=HTMLResponse)
async def new_person(
    request: Request,
    video_id: int,
    view_mode: str = Query(default="person", pattern="^(person|person_scene)$")
):
    """Person editor, creating a person of a view."""
    return templates.TemplateResponse(
        request=request, name="person_editor.html",
        context={
            "video_id": video_id,
            "person_id": None,
            "view_mode": view_mode,
            "automatic_face_count": DEFAULT_FACE_REFERENCE_COUNT
        }
    )


@app.get("/video/{video_id}/person/{person_id}", response_class=HTMLResponse)
async def edit_person(request: Request, video_id: int, person_id: int):
    """Person editor: frames, references, name and notes of a person."""
    return templates.TemplateResponse(
        request=request, name="person_editor.html",
        context={
            "video_id": video_id,
            "person_id": person_id,
            "view_mode": None,
            "automatic_face_count": DEFAULT_FACE_REFERENCE_COUNT
        }
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
