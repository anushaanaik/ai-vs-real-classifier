"""
routes.py — FastAPI route definitions for the AI vs Real classifier API.

Routes:
  POST /predict    → Upload image, get prediction JSON
  GET  /config     → Return model config/metadata
  GET  /health     → Liveness check
"""
import io
import os
import sys

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.predict import predict_image
from src.utils import get_logger, load_class_config

logger = get_logger(log_file="logs/app.log")

router = APIRouter()

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}
MAX_UPLOAD_MB = 10


# ─── Health ──────────────────────────────────────────────────────────────────

@router.get("/health")
async def health():
    return {"status": "ok"}


# ─── Model config ─────────────────────────────────────────────────────────────

@router.get("/config")
async def get_config():
    try:
        cfg = load_class_config("models/class_names.json")
        return JSONResponse(content=cfg)
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Model config not found. Run training first.",
        )


# ─── Prediction ───────────────────────────────────────────────────────────────

@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Upload an image and receive a prediction.

    Returns JSON:
    {
      "label":      "AI" | "REAL" | "UNCERTAIN",
      "confidence": 0.87,
      "ai_prob":    0.87,
      "real_prob":  0.13,
      "tta_steps":  8,
      "filename":   "photo.jpg"
    }
    """
    # Validate content type
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {file.content_type}. Use JPEG, PNG, or WebP.",
        )

    # Read & size-check
    contents = await file.read()
    size_mb = len(contents) / (1024 ** 2)
    if size_mb > MAX_UPLOAD_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f} MB). Max {MAX_UPLOAD_MB} MB.",
        )

    # Open as PIL
    try:
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot open image: {e}")

    # Run inference
    try:
        result = predict_image(
            pil_img,
            model_path="models/model_v1.keras",
            config_path="models/class_names.json",
            verbose=False,
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Model not found. Run `python src/train.py` first.",
        )
    except Exception as e:
        logger.error("Prediction error: %s", e)
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    result["filename"] = file.filename
    logger.info(
        "Prediction | file=%s | label=%s | confidence=%.4f",
        file.filename, result["label"], result["confidence"],
    )
    return JSONResponse(content=result)
