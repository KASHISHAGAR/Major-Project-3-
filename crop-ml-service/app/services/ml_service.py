"""
Wraps the trained crop-recommendation model package (model + scaler +
label map) saved from crop_recommendation_model.ipynb. Loaded once at
startup and reused for every request.
"""

import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd

from app.config import settings
from app.ml.crop_metadata import CROP_DISPLAY_NAMES


def _resolve(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        project_root = Path(__file__).resolve().parents[2]
        path = project_root / path_str
    return path


class MLService:
    def __init__(self) -> None:
        self._package: Optional[Dict[str, Any]] = None

    def load(self) -> None:
        with open(_resolve(settings.MODEL_PATH), "rb") as f:
            self._package = pickle.load(f)

    def is_loaded(self) -> bool:
        return self._package is not None

    @property
    def model_name(self) -> str:
        return self._package["model_name"] if self._package else ""

    @property
    def accuracy(self) -> float:
        return float(self._package["accuracy"]) if self._package else 0.0

    def predict(
        self,
        N: float,
        P: float,
        K: float,
        temperature: float,
        humidity: float,
        ph: float,
        rainfall: float,
        top_n: int = 5,
    ) -> List[Dict[str, Any]]:
        if self._package is None:
            raise RuntimeError("Model is not loaded")

        model = self._package["model"]
        scaler = self._package["scaler"]
        crop_dict_reverse = self._package["crop_dict_reverse"]
        feature_order = self._package["features"]

        row = pd.DataFrame(
            [[N, P, K, temperature, humidity, ph, rainfall]],
            columns=feature_order,
        )
        scaled = scaler.transform(row)
        probabilities = model.predict_proba(scaled)[0]
        classes = model.classes_

        ranked_indices = probabilities.argsort()[::-1][:top_n]

        results = []
        for idx in ranked_indices:
            label_id = int(classes[idx])
            crop_key = crop_dict_reverse[label_id]
            results.append(
                {
                    "crop": crop_key,
                    "display_name": CROP_DISPLAY_NAMES.get(crop_key, crop_key.title()),
                    "probability": round(float(probabilities[idx]), 4),
                }
            )
        return results


ml_service = MLService()
