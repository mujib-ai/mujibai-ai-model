# app/main.py

import json
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
    allow_origins=["https://www.mujibai.net",
                    "https://mujibai.net",], 
    allow_credentials=True,
    allow_methods=["POST"],
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
    await websocket.accept()
    session = RealtimeWhisperSession(model)

    try:
        while True:
            message = await websocket.receive()

            # AUDIO CHUNK
            if "bytes" in message:
                audio_chunk = message["bytes"]
                samples = decode_audio_bytes(audio_chunk)
                session.add_audio(samples)

                partial = session.transcribe_partial()
                if partial.strip():
                    clean = normalize_arabic(partial)
                    await websocket.send_json({"partial": clean})

            # CONTROL MESSAGE
            elif "text" in message:
                data = json.loads(message["text"])

                # CLIENT SIGNALS AUDIO IS DONE
                if data.get("event") == "end":
                    final_text = session.transcribe_final()
                    clean = normalize_arabic(final_text)

                    await websocket.send_json({
                        "final": clean
                    })
                    break

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected early.")

    except Exception:
        logger.exception("Unhandled websocket error")
        try:
            await websocket.send_json({"error": "internal server error"})
        except Exception:
            pass


# -----------------------------
# Documentation-only route
# -----------------------------
@app.get(
    "/ws/transcribe",
    tags=["WebSocket STT"],
    summary="Real-Time Speech-to-Text WebSocket (Full Documentation)",
    description="""
        # 🎙️ Real-Time Speech-to-Text WebSocket API

        This document describes the **real-time streaming Speech-to-Text (STT) WebSocket API**.

        > ⚠️ **Important**  
        > This endpoint is **documentation-only**.  
        > The actual transcription service runs over **WebSocket**, not HTTP.

        ---

        ## 1️⃣ What this WebSocket does

        The `/ws/transcribe` WebSocket provides **low-latency, real-time speech recognition**.

        - Audio is streamed **continuously** from client to server
        - Transcription results are streamed **incrementally** from server to client
        - The connection remains open for the duration of the session

        This design is optimized for:
        - Live captions
        - Voice assistants
        - Call center analytics
        - Conversational AI

        ---

        ## 2️⃣ WebSocket Endpoint

        ws://<HOST>:8000/ws/transcribe
        
        ---

        ## 3️⃣ Connection Lifecycle

        ### Step 1 — Client connects
        The client opens a WebSocket connection.
        
        ### Step 2 — Audio streaming begins
        The client sends **binary audio chunks** repeatedly.


        ### Step 3 — Partial transcription returned
        The server sends **partial transcription hypotheses** as JSON.


        ### Step 4 — Session continues
        Steps 2 and 3 repeat until the client disconnects.

        ### Step 5 — Connection closes
        The session ends when:
        - Client closes the WebSocket
        - Network disconnects
        - Server encounters an unrecoverable error

        ---

        ## 4️⃣ Audio Format Requirements (MANDATORY)

        All audio sent to the server **must** conform to the following:

        | Parameter | Requirement |
        |--------|-------------|
        | Sample Rate | **16,000 Hz** |
        | Channels | **Mono (1 channel)** |
        | Encoding | **PCM 16-bit signed (int16)** |
        | Endianness | Little-endian |
        | Container | Raw PCM or WAV |
        | Language | Arabic (`ar`) |

        ❌ Stereo audio is **not supported**  
        ❌ Non-16kHz audio will produce incorrect results

        ---

        ## 5️⃣ Audio Chunking Guidelines

        | Property | Recommended |
        |-------|------------|
        | Chunk Duration | 300–1000 ms |
        | Chunk Size | 4,800 – 16,000 samples |
        | Send Interval | Real-time (sleep between sends) |

        Sending very small chunks (<100 ms) may:
        - Increase latency
        - Reduce transcription quality

        ---

        ## 6️⃣ Client → Server Messages

        ### Message Type
        - **Binary WebSocket frame**

        ### Payload

        ### Example
        ```python
        ws.send(binary_audio_chunk)

        {
        "partial": "اتنهد وقال الرجل بصوت منخفض"
        }


            """
)
def websocket_transcribe_docs():
    return {
        "note": "This endpoint is documentation-only. See description for usage."
    }