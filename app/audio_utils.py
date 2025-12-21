# app/audio_utils.py
import numpy as np
import soundfile as sf
import io
import logging
from app.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

def decode_audio_bytes(audio_bytes: bytes):
    """
    Converts audio bytes into float32 samples at 16 kHz mono.
    Clients must stream raw WAV or PCM chunks.
    """
    try:
        data, sr = sf.read(io.BytesIO(audio_bytes))
        if len(data.shape) == 2:  # stereo → mono
            data = np.mean(data, axis=1)
        if sr != 16000:
            raise ValueError("Expected 16kHz audio. Please resample on client side.")
        return data.astype("float32")
    except Exception:
        # If it's raw PCM chunk (int16)
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
        logger.warning("Audio decode failed, trying raw PCM: %s", Exception)
        return audio / 32768.0
