import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routers import parse

# Configure logging so our logs show up
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-12s | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger("ocr.app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-load all OCR models at startup so first request isn't slow
    logger.info("Pre-loading OCR models...")
    from app.services.ocr import OCREngine
    OCREngine.get()
    logger.info("OCR models ready — accepting requests")
    yield


app = FastAPI(title="Korean Document OCR", version="0.2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(parse.router, prefix="/api")


@app.get("/api/health")
async def health():
    return {"status": "ok"}
