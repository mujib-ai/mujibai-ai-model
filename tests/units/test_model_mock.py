from unittest.mock import patch
from app.model_loader import load_whisper_model

@patch("app.model_loader.WhisperModel")
def test_model_loader(mock_whisper):
    mock_whisper.return_value = "mock_model"
    model = load_whisper_model()
    assert model == "mock_model"
