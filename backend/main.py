"""Application entrypoint.

    uvicorn backend.main:app --reload
    python -m backend.main

Startup work (seeding the vector store, embedding topic anchors) is best-effort:
if Ollama or Qdrant is not up yet the server still boots and reports the problem
on /api/health rather than refusing to start.
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend import config
from backend.api.routes import router
from backend.scoring.intent_analyzer import aprecompute_topic_anchors, validate_anchors
from backend.vector_store.seeder import seed_mentor_knowledge

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("backend")

FRONTEND_DIR = config.PROJECT_ROOT / "frontend"


async def _warm_up() -> None:
    """Seed retrieval and embed topic anchors. Never fatal."""
    if config.SEED_ON_STARTUP:
        try:
            logger.info("Seeding mentor knowledge into Qdrant...")
            # Seeding is blocking (HTTP embed calls plus disk writes); run it off
            # the event loop so the server can start serving immediately after.
            await asyncio.to_thread(seed_mentor_knowledge)
        except Exception as exc:
            logger.warning(
                "Seeding skipped: %s. Retrieval will be unavailable until you run: "
                "python -m backend.vector_store.seeder",
                exc,
            )
    try:
        logger.info("Precomputing topic anchor vectors...")
        await aprecompute_topic_anchors()
    except Exception as exc:
        logger.warning(
            "Could not precompute topic anchors: %s. They will be computed on the "
            "first request instead.",
            exc,
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    config.validate()
    validate_anchors()
    await _warm_up()
    logger.info("Ready on http://%s:%d", config.HOST, config.PORT)
    yield


app = FastAPI(
    title="AI Wisdom Router",
    version="1.0.0",
    description="Routes life questions to a weighted council of mentor personas.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    # Credentials cannot be combined with a wildcard origin; browsers reject it.
    allow_credentials=config.CORS_ORIGINS != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")

if FRONTEND_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_frontend() -> FileResponse:
        return FileResponse(str(FRONTEND_DIR / "index.html"))

else:  # pragma: no cover - only hit in a backend-only deployment
    logger.warning("Frontend directory not found at %s; API only.", FRONTEND_DIR)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=config.HOST,
        port=config.PORT,
        reload=os.getenv("RELOAD", "true").lower() in ("1", "true", "yes", "on"),
    )
