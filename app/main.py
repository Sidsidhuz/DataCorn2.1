"""
app/main.py
=============

FastAPI application entrypoint.
Registers routes, sets up database lifespan management, and configures CORS middleware.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from app.config import settings
from app.database import init_db
from app.routers import datasets, training, models, reports


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once when the server starts: ensure local data folders exist and
    # the SQLite schema is created.
    init_db()

    # --- Startup cleanup: heal stale jobs from a previous server crash/restart ---
    # If the server was killed mid-training, jobs are stuck as "queued" or "running"
    # in the DB but their background threads are gone. Mark them as "failed" so
    # users can immediately start a new training run without hitting a 409 Conflict.
    from app.database import SessionLocal
    from app.models.training_job import TrainingJob
    with SessionLocal() as db:
        stale_jobs = (
            db.query(TrainingJob)
            .filter(TrainingJob.status.in_(["queued", "running"]))
            .all()
        )
        if stale_jobs:
            for job in stale_jobs:
                job.status = "failed"
                job.error_message = "Server restarted while job was in progress. Please start a new training run."
            db.commit()
            import logging
            logging.getLogger("uvicorn").warning(
                f"[DataCorn] Marked {len(stale_jobs)} stale training job(s) as 'failed' on startup."
            )

    yield
    # (nothing to clean up on shutdown for a local SQLite app)



app = FastAPI(
    title="DataCorn API",
    description="Local AutoML & Explainable AI Platform — backend API.",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS configuration for local development connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins locally
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers under /api/v1
app.include_router(datasets.router, prefix="/api/v1")
app.include_router(training.router, prefix="/api/v1")
app.include_router(models.router, prefix="/api/v1")
app.include_router(reports.router, prefix="/api/v1")


@app.get("/health", tags=["meta"])
def health_check() -> dict:
    """Simple liveness check, also useful for confirming config is loaded."""
    return {
        "status": "ok",
        "database_url": settings.database_url,
        "data_dir": str(settings.data_path),
    }


@app.get("/", include_in_schema=False)
def root_redirect():
    """Redirect root to interactive API documentation."""
    return RedirectResponse(url="/docs")
