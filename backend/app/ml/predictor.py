"""
Inference predictor — loads the trained model once at startup and serves
sub-second predictions for uploaded images.

The model is loaded as a module-level singleton to avoid loading it on
every request, keeping inference latency under 1 second.
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass
from typing import List

import torch
from PIL import Image

from app.core.config import settings
from app.ml.model import SkinAnomalyCNN, load_model
from app.ml.transforms import get_inference_transforms

# ---------------------------------------------------------------------------
# Module-level singleton — loaded once at app startup
# ---------------------------------------------------------------------------
_model: SkinAnomalyCNN | None = None
_transform = get_inference_transforms()


def get_model() -> SkinAnomalyCNN:
    """Lazy-load the model on first call; return cached instance thereafter."""
    global _model
    if _model is None:
        _model = load_model(settings.MODEL_PATH, num_classes=settings.NUM_CLASSES)
    return _model


# ---------------------------------------------------------------------------
# Prediction result dataclass
# ---------------------------------------------------------------------------
@dataclass
class PredictionResult:
    predicted_class: str
    predicted_class_index: int
    confidence: float
    risk_level: str
    all_probabilities: List[dict]
    inference_time_ms: float
    low_confidence_warning: bool


# ---------------------------------------------------------------------------
# Core predict function
# ---------------------------------------------------------------------------
def predict_image(image_bytes: bytes) -> PredictionResult:
    """
    Run inference on raw image bytes.

    Steps:
        1. Decode bytes to PIL Image
        2. Apply inference transforms (resize 224x224, normalize)
        3. Forward pass through SkinAnomalyCNN
        4. Return top prediction with confidence and all class probabilities

    Raises:
        ValueError: if image cannot be decoded
    """
    t0 = time.perf_counter()

    # Decode image
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise ValueError(f"Cannot decode image: {e}") from e

    # Preprocess
    tensor = _transform(image).unsqueeze(0)  # (1, 3, 224, 224)

    # Inference
    model = get_model()
    with torch.no_grad():
        probs = model.predict_proba(tensor).squeeze(0)  # (num_classes,)

    elapsed_ms = (time.perf_counter() - t0) * 1000

    # Build result
    top_idx = int(probs.argmax())
    top_confidence = float(probs[top_idx])

    all_probs = [
        {
            "class": settings.CLASS_LABELS[i],
            "probability": float(probs[i]),
            "risk": settings.CLASS_RISK[i],
        }
        for i in range(len(settings.CLASS_LABELS))
    ]
    # Sort by probability descending
    all_probs.sort(key=lambda x: x["probability"], reverse=True)

    return PredictionResult(
        predicted_class=settings.CLASS_LABELS[top_idx],
        predicted_class_index=top_idx,
        confidence=top_confidence,
        risk_level=settings.CLASS_RISK[top_idx],
        all_probabilities=all_probs,
        inference_time_ms=round(elapsed_ms, 2),
        low_confidence_warning=top_confidence < settings.CONFIDENCE_THRESHOLD,
    )
