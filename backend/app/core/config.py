"""Application configuration."""

from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    APP_NAME: str = "Primary Care Triage Assistant"
    DEBUG: bool = False

    # Model
    MODEL_PATH: str = "app/ml/weights/skin_cnn.pt"
    NUM_CLASSES: int = 8
    IMAGE_SIZE: int = 224
    CONFIDENCE_THRESHOLD: float = 0.5

    # CORS
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:5173"]

    # Class labels
    CLASS_LABELS: List[str] = [
        "Benign keratosis",
        "Melanocytic nevi",
        "Melanoma",
        "Basal cell carcinoma",
        "Actinic keratosis",
        "Vascular lesion",
        "Dermatofibroma",
        "Normal skin",
    ]

    # Risk levels per class (for clinical urgency display)
    CLASS_RISK: List[str] = [
        "low",
        "low",
        "high",
        "high",
        "medium",
        "medium",
        "low",
        "none",
    ]


settings = Settings()
