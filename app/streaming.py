# app/streaming.py

import numpy as np
import logging
from app.logging_config import setup_logging

logger = logging.getLogger(__name__)
setup_logging()

class RealtimeWhisperSession:
    def __init__(self, model):
        self.model = model

        # Buffer for real-time partials (short, sliding)
        self.rolling_buffer = np.zeros(0, dtype=np.float32)

        # Buffer for final transcription (entire audio)
        self.full_buffer = np.zeros(0, dtype=np.float32)

        # 3 seconds rolling window
        self.window_size = 16000 * 3

    def add_audio(self, samples: np.ndarray):
        # Keep everything for final transcription
        self.full_buffer = np.concatenate([self.full_buffer, samples])

        # Keep only recent audio for partials
        self.rolling_buffer = np.concatenate([self.rolling_buffer, samples])
        if len(self.rolling_buffer) > self.window_size:
            self.rolling_buffer = self.rolling_buffer[-self.window_size:]

    def transcribe_partial(self) -> str:
        """
        Fast, low-latency partial transcription.
        """
        if len(self.rolling_buffer) < 16000:
            return ""

        segments, _ = self.model.transcribe(
            self.rolling_buffer,
            language="ar",
            beam_size=1,
            vad_filter=True
        )

        return " ".join(s.text for s in segments).strip()

    def transcribe_final(self) -> str:
        """
        One clean final transcription pass on the full audio.
        """
        if len(self.full_buffer) < 16000:
            return ""

        segments, _ = self.model.transcribe(
            self.full_buffer,
            language="ar",
            beam_size=3,
            vad_filter=True
        )

        return " ".join(s.text for s in segments).strip()
