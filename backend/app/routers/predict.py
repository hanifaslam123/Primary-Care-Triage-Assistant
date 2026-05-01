"""
Prediction router — POST /api/predict

Accepts an uploaded skin image and returns classification results
from the PyTorch CNN model with < 1 second response time.
"""

from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from pydantic import BaseModel

from app.ml.predictor import PredictionResult, predict_image

router = APIRouter()

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
MAX_FILE_SIZE_MB = 10


class ClassProbability(BaseModel):
    class_name: str
    probability: float
    risk: str


class PredictionResponse(BaseModel):
    predicted_class: str
    predicted_class_index: int
    confidence: float
    risk_level: str
    all_probabilities: List[dict]
    inference_time_ms: float
    low_confidence_warning: bool
    disclaimer: str


DISCLAIMER = (
    "This tool is intended to assist healthcare professionals and is NOT a "
    "substitute for professional medical diagnosis. All results must be "
    "reviewed by a licensed healthcare provider."
)


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Classify skin anomaly image",
    description=(
        "Upload a dermatoscopic image (JPG/PNG/WebP, max 10MB). "
        "Returns predicted skin anomaly class, confidence score, "
        "risk level, and full probability distribution. "
        "Inference time is typically under 1 second."
    ),
)
async def classify_skin_image(
    file: UploadFile = File(..., description="Skin image file (JPG/PNG/WebP)"),
):
    """Run inference on an uploaded skin image and return classification results."""

    # Validate content type
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported image type '{file.content_type}'. Use JPG, PNG, or WebP.",
        )

    # Read and validate file size
    image_bytes = await file.read()
    size_mb = len(image_bytes) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large ({size_mb:.1f} MB). Maximum allowed: {MAX_FILE_SIZE_MB} MB.",
        )

    # Run inference
    try:
        result: PredictionResult = predict_image(image_bytes)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )

    return PredictionResponse(
        predicted_class=result.predicted_class,
        predicted_class_index=result.predicted_class_index,
        confidence=result.confidence,
        risk_level=result.risk_level,
        all_probabilities=result.all_probabilities,
        inference_time_ms=result.inference_time_ms,
        low_confidence_warning=result.low_confidence_warning,
        disclaimer=DISCLAIMER,
    )
