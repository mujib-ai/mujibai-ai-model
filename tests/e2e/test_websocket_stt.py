import pytest
import numpy as np
from fastapi.testclient import TestClient
from app.main import app
import app.main as main

client = TestClient(app)

def test_websocket_stt(mocker):
    # Mock Whisper model to avoid GPU during tests
    mock_model = mocker.Mock()
    mock_model.transcribe.return_value = (
        [type("seg", (), {"text": "hello"})()],
        None
    )

    # Replace the loaded model in the main module so the websocket handler uses it
    main.model = mock_model
    app.state._model = mock_model

    with client.websocket_connect("/ws/transcribe") as websocket:
        # Send 1 sec of silence
        chunk = (np.zeros(16000, dtype=np.int16)).tobytes()
        websocket.send_bytes(chunk)

        data = websocket.receive_json()
        assert "partial" in data
        assert data["partial"] == "hello"
