import numpy as np
from app.streaming import RealtimeWhisperSession

class MockModel:
    def transcribe(self, *_args, **_kwargs):
        return ([type("seg", (), {"text": "test"})()], None)

def test_streaming_session():
    session = RealtimeWhisperSession(MockModel())

    chunk = np.ones(16000, dtype=np.float32)
    session.add_audio(chunk)

    assert len(session.buffer) == 16000
