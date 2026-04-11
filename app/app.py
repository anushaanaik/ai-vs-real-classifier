"""
app.py — FastAPI backend for the AI vs Real image classifier.

Endpoints:
  GET  /               → Health check
  POST /predict        → JSON body (Base64 image) → prediction JSON
  POST /predict/upload → multipart file upload    → prediction JSON
  GET  /config         → Model configuration info
  GET  /health         → Liveness probe

Start with:
  uvicorn app.app:app --reload --port 8000
"""
import os
import sys
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.routes import router
from src.utils import get_logger

logger = get_logger(log_file="logs/app.log")

app = FastAPI(
    title="AI vs Real Image Classifier",
    description=(
        "Detect whether an image is AI-generated or a real photograph "
        "using EfficientNetV2-S with Test-Time Augmentation (TTA)."
    ),
    version="1.0.0",
)

# CORS — allow the local frontend / any origin to call the API
# Tighten allow_origins in production (e.g. ["http://localhost:3000"])
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.on_event("startup")
async def startup_event():
    logger.info("AI vs Real Classifier API starting up...")
    # Warm up: load model once so first request isn't slow
    try:
        from src.predict import _load_model, _load_config
        _load_config()
        _load_model()
        logger.info("Model loaded successfully and ready to serve.")
    except Exception as e:
        logger.warning(
            "Model not yet available (run `python pipeline/pipeline.py` first): %s", e
        )


@app.get("/")
async def root():
    return {
        "status": "ok",
        "service": "AI vs Real Image Classifier",
        "version": "1.0.0",
        "docs": "/docs",
    }
