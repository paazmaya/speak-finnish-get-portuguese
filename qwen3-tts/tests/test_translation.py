"""
Tests for the translation pipeline.
"""

import pytest
import numpy as np
from pathlib import Path

from qwen3_tts_translator.audio import (
    resample_audio,
    AudioRecorder,
)
from qwen3_tts_translator.translation import TextTranslator


class TestAudio:
    """Test audio processing functions."""

    def test_resample_audio_no_change(self):
        """Test resampling with same sample rate."""
        audio = np.random.randn(16000).astype(np.float32)
        resampled = resample_audio(audio, 16000, 16000)
        assert len(resampled) == len(audio)

    def test_resample_audio_upsample(self):
        """Test upsampling audio."""
        audio = np.random.randn(8000).astype(np.float32)
        resampled = resample_audio(audio, 8000, 16000)
        assert len(resampled) == 16000

    def test_resample_audio_downsample(self):
        """Test downsampling audio."""
        audio = np.random.randn(16000).astype(np.float32)
        resampled = resample_audio(audio, 16000, 8000)
        assert len(resampled) == 8000

    def test_recorder_init(self):
        """Test AudioRecorder initialization."""
        recorder = AudioRecorder(sample_rate=16000)
        assert recorder.sample_rate == 16000
        assert not recorder.is_recording
        assert len(recorder.frames) == 0


class TestTranslation:
    """Test translation functions."""

    @pytest.fixture
    def translator(self):
        """Create translator instance (using CPU)."""
        return TextTranslator(device="cpu")

    def test_translate_fi_to_en(self, translator):
        """Test Finnish to English translation."""
        result = translator.translate_fi_to_en("Hei maailma")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_translate_en_to_pt(self, translator):
        """Test English to Portuguese translation."""
        result = translator.translate_en_to_pt("Hello world")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_full_translation(self, translator):
        """Test full Finnish to Portuguese translation."""
        result = translator.translate("Hei maailma", verbose=False)
        assert isinstance(result, str)
        assert len(result) > 0


def test_import():
    """Test package can be imported."""
    import qwen3_tts_translator

    assert qwen3_tts_translator.__version__ == "0.1.0"
