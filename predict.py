"""
Prediction module for the Email/SMS Spam Detection system.

The model and vectoriser are loaded **once** when this module is first
imported.  Subsequent calls to ``predict()`` reuse them.
"""

import logging
import os
import pickle

import numpy as np

from data_preprocessing import transform_text

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")

# ---------------------------------------------------------------------------
# Load model and vectoriser once
# ---------------------------------------------------------------------------

def _load_pickle(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required file not found: {path}. "
            "Please run 'python train.py' first."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


_model = _load_pickle(MODEL_PATH)
_vectorizer = _load_pickle(VECTORIZER_PATH)

logger.info("Model and vectorizer loaded successfully.")

# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------

_LABEL_MAP = {0: "ham", 1: "spam"}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

MAX_INPUT_LENGTH = 10_000  # characters


def predict(text: str) -> dict:
    """Classify a single text as *spam* or *ham*.

    Parameters
    ----------
    text : str
        Raw email / SMS body.

    Returns
    -------
    dict
        ``{"label": "spam" | "ham", "confidence": float}``

    Raises
    ------
    ValueError
        If *text* is empty or exceeds ``MAX_INPUT_LENGTH``.
    """
    if not text or not text.strip():
        raise ValueError("Input text must not be empty.")

    if len(text) > MAX_INPUT_LENGTH:
        raise ValueError(
            f"Input text exceeds maximum length of {MAX_INPUT_LENGTH} characters."
        )

    # 1. Preprocess (same pipeline as training)
    cleaned = transform_text(text)

    # 2. Vectorise
    vector = _vectorizer.transform([cleaned])

    # 3. Predict
    prediction = _model.predict(vector)[0]
    label = _LABEL_MAP.get(prediction, "unknown")

    # 4. Confidence
    if hasattr(_model, "predict_proba"):
        probabilities = _model.predict_proba(vector)[0]
        confidence = float(np.max(probabilities))
    else:
        # Fallback for models that don't support predict_proba
        confidence = 1.0

    return {"label": label, "confidence": confidence}
