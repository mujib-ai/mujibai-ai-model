# app/model_loader.py

from faster_whisper import WhisperModel
from app.logging_config import setup_logging

import logging

setup_logging()
logger = logging.getLogger(__name__)

MODEL_SIZE = "large-v3_v1" # Switch to large-v3 if you want to test
DEVICE = "cuda"
COMPUTE_TYPE = "float16"

logger.info("Loading Whisper model: %s", MODEL_SIZE)

def load_whisper_model():
    try:
        model = WhisperModel(
            MODEL_SIZE,
            device=DEVICE,
            compute_type=COMPUTE_TYPE
        )
        print("Loaded WhisperModel on GPU.")
    except:
        print("GPU failed. Falling back to CPU int8.")
        logger.exception("Failed to load model on GPU. Falling back to CPU.")
        model = WhisperModel(
            MODEL_SIZE,
            device="cpu",
            compute_type="int8"
        )
    return model