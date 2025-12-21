import numpy as np
from app.audio_utils import decode_audio_bytes

def test_decode_raw_pcm():
    # 1 second of silence, int16
    raw = (np.zeros(16000, dtype=np.int16)).tobytes()
    samples = decode_audio_bytes(raw)

    assert isinstance(samples, np.ndarray)
    assert samples.shape[0] == 16000
    assert samples.dtype == np.float32
