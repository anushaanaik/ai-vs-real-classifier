"""
predict.py — Single-image inference with Test-Time Augmentation (TTA).

Usage (CLI):
  python src/predict.py --image path/to/image.jpg
  python src/predict.py --image path/to/image.jpg --tta 1   # no TTA (faster)

Returns JSON:
  {"label": "AI", "confidence": 0.87, "ai_prob": 0.87, "real_prob": 0.13}
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

# ─── Lazy model loading (singleton) ─────────────────────────────────────────

_model = None
_config: dict | None = None

DEFAULT_MODEL_PATH = "models/model_v1.keras"
DEFAULT_CONFIG_PATH = "models/class_names.json"


def _load_model(model_path: str = DEFAULT_MODEL_PATH):
    global _model
    if _model is None:
        import tensorflow as tf
        _model = tf.keras.models.load_model(model_path)
    return _model


def _load_config(config_path: str = DEFAULT_CONFIG_PATH) -> dict:
    global _config
    if _config is None:
        with open(config_path, "r") as f:
            _config = json.load(f)
    return _config


# ─── Core inference ──────────────────────────────────────────────────────────

def predict_image(
    image_source,
    model_path: str = DEFAULT_MODEL_PATH,
    config_path: str = DEFAULT_CONFIG_PATH,
    tta_steps: int | None = None,
    threshold: float | None = None,
    verbose: bool = True,
) -> dict:
    """
    Classify a single image as AI-generated or Real.

    Parameters
    ----------
    image_source : str | PIL.Image
        File path or PIL Image object.
    model_path : str
        Path to the saved Keras model.
    config_path : str
        Path to class_names.json.
    tta_steps : int | None
        Number of TTA passes. None → use config default (8).
    threshold : float | None
        Confidence threshold. None → use config default (0.60).
    verbose : bool
        Print result to console.

    Returns
    -------
    dict: {label, confidence, ai_prob, real_prob, tta_steps}
          label is 'AI' | 'REAL' | 'UNCERTAIN'
    """
    from src.preprocess import preprocess_image

    cfg = _load_config(config_path)
    model = _load_model(model_path)

    img_size = cfg.get("input_size", 224)
    tta = tta_steps if tta_steps is not None else cfg.get("tta_steps_default", 8)
    thresh = threshold if threshold is not None else cfg.get("confidence_threshold", 0.60)

    probs = []
    for i in range(tta):
        arr = preprocess_image(image_source, img_size=img_size, apply_aug=(i > 0))
        arr = np.expand_dims(arr, axis=0)
        prob = float(model.predict(arr, verbose=0)[0][0])
        probs.append(prob)

    real_prob = float(np.mean(probs))
    ai_prob = 1.0 - real_prob

    if real_prob >= thresh:
        label, confidence = "REAL", real_prob
    elif ai_prob >= thresh:
        label, confidence = "AI", ai_prob
    else:
        label, confidence = "UNCERTAIN", max(real_prob, ai_prob)

    result = {
        "label": label,
        "confidence": round(confidence, 4),
        "ai_prob": round(ai_prob, 4),
        "real_prob": round(real_prob, 4),
        "tta_steps": tta,
    }

    if verbose:
        emoji = {"AI": "🤖", "REAL": "📷", "UNCERTAIN": "❓"}[label]
        print(f"\n{emoji}  {label}  (confidence: {confidence:.1%})")
        print(f"   AI prob  : {ai_prob:.1%}")
        print(f"   Real prob: {real_prob:.1%}")
        print(f"   TTA steps: {tta}")
        if label == "UNCERTAIN":
            print("   ⚠️  Below threshold — treat with caution")

    return result


# ─── Batch inference ────────────────────────────────────────────────────────

def predict_batch(
    image_paths: list[str],
    model_path: str = DEFAULT_MODEL_PATH,
    config_path: str = DEFAULT_CONFIG_PATH,
    tta_steps: int = 1,
    threshold: float | None = None,
) -> list[dict]:
    """
    Run inference on a list of image paths.
    Uses tta_steps=1 by default for speed; increase for accuracy.
    """
    results = []
    for path in image_paths:
        r = predict_image(
            path,
            model_path=model_path,
            config_path=config_path,
            tta_steps=tta_steps,
            threshold=threshold,
            verbose=False,
        )
        r["path"] = path
        results.append(r)
    return results


# ─── CLI entry point ─────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="AI vs Real — single image prediction")
    p.add_argument("--image", required=True, help="Path to image file")
    p.add_argument("--model", default=DEFAULT_MODEL_PATH)
    p.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    p.add_argument("--tta", type=int, default=None, help="TTA steps (default: from config)")
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--json", action="store_true", help="Output raw JSON")
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")

    result = predict_image(
        args.image,
        model_path=args.model,
        config_path=args.config,
        tta_steps=args.tta,
        threshold=args.threshold,
        verbose=not args.json,
    )

    if args.json:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
