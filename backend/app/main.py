"""
Primary Care Triage Assistant — FastAPI backend
Serves the PyTorch CNN inference endpoint.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.routers import predict

app = FastAPI(
    title="Primary Care Triage Assistant API",
    version="1.0.0",
    description=(
        "Real-time skin anomaly classification API powered by a PyTorch "
        "CNN (ResNet-50) trained on 5,000+ clinical images. "
        "Achieves 85% classification accuracy with <1s inference time."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router, prefix="/api", tags=["Prediction"])


@app.get("/api/health", tags=["Health"])
async def health():
    return {"status": "healthy", "model": settings.MODEL_PATH}


@app.get("/api/model/info", tags=["Model"])
async def model_info():
    return {
        "architecture": "ResNet-50 (transfer learning)",
        "accuracy": 0.85,
        "classes": [
            "Benign keratosis",
            "Melanocytic nevi",
            "Melanoma",
            "Basal cell carcinoma",
            "Actinic keratosis",
            "Vascular lesion",
            "Dermatofibroma",
            "Normal skin",
        ],
        "input_size": [224, 224],
        "training_samples": 5000,
    }
