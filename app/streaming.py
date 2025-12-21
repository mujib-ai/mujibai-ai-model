# app/streaming.py

import numpy as np
import logging
from app.logging_config import setup_logging

logger = logging.getLogger(__name__)
setup_logging()

class RealtimeWhisperSession:
    def __init__(self, model):
        self.model = model
        self.buffer = np.zeros(0, dtype=np.float32)
        self.window_size = 16000 * 3     # 3 seconds window TODO: make configurable

    def add_audio(self, samples):
        self.buffer = np.concatenate([self.buffer, samples])
        # keep last N seconds only
        if len(self.buffer) > self.window_size:
            self.buffer = self.buffer[-self.window_size:]

    def transcribe_partial(self):
        """
        Transcribe last 2–3 seconds.
        Avoids long delays & gives real-time partials.
        """
        if len(self.buffer) < 16000:  # less than 1 sec
            logger.warning("Insufficient audio for transcription")
            return ""

        # Run whisper in non-blocking way on recent chunk
        # **NOTE: This call is synchronous and will block the event loop.
        # TODO: Move transcription to a thread pool / worker for production.
        segments, _ = self.model.transcribe(
            self.buffer,
            language="ar",
            beam_size=1,
            vad_filter=True
        )
        text = " ".join(s.text for s in segments)
        return text.strip()
