import json
import os
from pathlib import Path
from typing import Iterable
import numpy as np
from PIL import Image

from .config import (
    CLASS_NAMES_PATH,
    CONFUSION_MATRIX_PATH,
    HISTORY_PLOT_PATH,
    MODELS_DIR,
    OUTPUT_DIR,
)


def ensure_directories() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_class_names(class_names: Iterable[str]) -> None:
    ensure_directories()
    with open(CLASS_NAMES_PATH, "w", encoding="utf-8") as f:
        json.dump(list(class_names), f, indent=2)


# 🔥 FIXED FUNCTION (IMPORTANT)
def load_class_names(path=None) -> list[str]:
    """
    Loads class names from JSON file.
    Works both for training (config path) and deployment (custom path).
    """
    if path is None:
        path = CLASS_NAMES_PATH

    # Convert Path → string if needed
    path = str(path)

    if not os.path.exists(path):
        raise FileNotFoundError(f"class_names.json not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def preprocess_pil_image(image: Image.Image, img_size: tuple[int, int]) -> np.ndarray:
    image = image.convert("RGB")
    image = image.resize(img_size)
    array = np.array(image, dtype=np.float32)
    array = np.expand_dims(array, axis=0)
    return array


def merge_histories(history_a: dict, history_b: dict) -> dict:
    keys = set(history_a.keys()) | set(history_b.keys())
    merged = {}
    for key in keys:
        merged[key] = history_a.get(key, []) + history_b.get(key, [])
    return merged


def plot_training_history(history: dict) -> Path:
    ensure_directories()
    # Plot generation was intentionally removed.
    return HISTORY_PLOT_PATH


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str]) -> Path:
    ensure_directories()
    # Plot generation was intentionally removed.
    return CONFUSION_MATRIX_PATH
