# app/main.py

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from app.model_loader import load_whisper_model
from app.normalizer import normalize_arabic
from app.audio_utils import decode_audio_bytes
from app.streaming import RealtimeWhisperSession
from app.logging_config import setup_logging

import logging

# initialize logging first
setup_logging()
logger = logging.getLogger(__name__)
logger.info("app.main loaded")

app = FastAPI()

# -----------------------------
# CORS (Authentication)
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Load the Whisper model once
# -----------------------------
model = load_whisper_model()


# -----------------------------
# HEALTH CHECK
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok", "model": "whisper-faster"}


# -----------------------------
# REAL-TIME STREAMING STT
# -----------------------------
@app.websocket("/ws/transcribe")
async def websocket_endpoint(websocket: WebSocket):
    """
    Real-time speech-to-text streaming.
    Send small PCM/WAV chunks repeatedly.
    This code decodes and feed to FasterWhisper and send partial transcription back back.
    """
    await websocket.accept()
    session = RealtimeWhisperSession(model)

    try:
        while True:
            # Receive raw audio bytes (chunk)
            audio_chunk = await websocket.receive_bytes()

            # Convert bytes to float32 waveform
            samples = decode_audio_bytes(audio_chunk)
            session.add_audio(samples)

            # Get partial transcription
            text = session.transcribe_partial()

            if text.strip():
                clean = normalize_arabic(text)
                await websocket.send_json({"partial": clean})

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected.")
    except Exception:
        logger.exception("Unhandled websocket error")
            # Attempt to notify client of error
        try:
            await websocket.send_json({"error": "internal server error"})
        except Exception:
            logger.debug("Failed to send error to client.")