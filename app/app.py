"""
app.py — FastAPI backend for the AI vs Real image classifier.

Endpoints:
  GET  /           → Health check
  POST /predict    → Upload image → JSON prediction
  GET  /config     → Model configuration info

Start with:
  uvicorn app.app:app --reload --port 8000
"""
import io
import os
import sys
import logging

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.routes import router
from src.utils import get_logger

logger = get_logger(log_file="logs/app.log")

app = FastAPI(
    title="AI vs Real Image Classifier",
    description="Detect whether an image is AI-generated or real using EfficientNetB3.",
    version="1.0.0",
)

# CORS — allow the local frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.on_event("startup")
async def startup_event():
    logger.info("AI vs Real Classifier API starting up...")
    # Warm up the model by loading it once
    try:
        from src.predict import _load_model, _load_config
        _load_config()
        _load_model()
        logger.info("Model loaded and ready.")
    except Exception as e:
        logger.warning("Model not yet available (run training first): %s", e)


@app.get("/")
async def root():
    return {"status": "ok", "service": "AI vs Real Image Classifier", "version": "1.0.0"}
