"""Integration tests for speech translation."""

import pytest
import tempfile
from pathlib import Path
import soundfile as sf
import torch

from seamless_translator import SeamlessTranslator


@pytest.fixture
def translator():
    """Create a translator instance for testing."""
    # Use CPU for testing to avoid GPU memory issues in CI
    return SeamlessTranslator(device="cpu", load_in_8bit=False)


def test_terve_translation(translator, terve_audio_path, tmp_path):
    """
    Test translation of Finnish word 'Terve' to Portuguese.
    
    Expected translation: 'Olá' or 'Oi' (Portuguese for 'Hello')
    """
    # Skip if test audio doesn't exist
    if not terve_audio_path.exists():
        pytest.skip(f"Test audio file not found: {terve_audio_path}")
    
    # Verify input audio exists and is readable
    audio_data, sample_rate = sf.read(str(terve_audio_path))
    assert len(audio_data) > 0, "Input audio is empty"
    assert sample_rate > 0, "Invalid sample rate"
    
    # Setup output path
    output_path = tmp_path / "terve_translated.wav"
    
    # Perform translation
    translator.translate(
        input_audio_path=str(terve_audio_path),
        output_audio_path=str(output_path),
        src_lang="fin",
        tgt_lang="por",
        verbose=True,
    )
    
    # Verify output audio was created
    assert output_path.exists(), "Output audio file was not created"
    
    # Verify output audio has content
    output_data, output_sr = sf.read(str(output_path))
    assert len(output_data) > 0, "Output audio is empty"
    assert output_sr > 0, "Invalid output sample rate"
    
    # Verify output has reasonable duration (at least 0.1 seconds)
    duration = len(output_data) / output_sr
    assert duration >= 0.1, f"Output audio too short: {duration:.3f}s"
    assert duration <= 10.0, f"Output audio too long: {duration:.3f}s"
    
    print(f"✓ Translation successful:")
    print(f"  Input:  {len(audio_data)/sample_rate:.2f}s at {sample_rate} Hz")
    print(f"  Output: {duration:.2f}s at {output_sr} Hz")


def test_translation_with_nonexistent_file(translator, tmp_path):
    """Test that translation fails gracefully with nonexistent input."""
    input_path = tmp_path / "nonexistent.wav"
    output_path = tmp_path / "output.wav"
    
    with pytest.raises(Exception):
        translator.translate(
            input_audio_path=str(input_path),
            output_audio_path=str(output_path),
            src_lang="fin",
            tgt_lang="por",
        )


def test_translator_initialization():
    """Test that translator can be initialized with different configs."""
    # CPU initialization
    translator_cpu = SeamlessTranslator(device="cpu", load_in_8bit=False)
    assert translator_cpu.device == "cpu"
    
    # Auto device detection
    translator_auto = SeamlessTranslator(load_in_8bit=False)
    assert translator_auto.device in ["cpu", "cuda", "mps"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_translator_cuda():
    """Test translator with CUDA if available."""
    translator = SeamlessTranslator(device="cuda", load_in_8bit=True)
    assert translator.device == "cuda"


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_translator_mps():
    """Test translator with MPS (Apple Silicon) if available."""
    translator = SeamlessTranslator(device="mps", load_in_8bit=False)
    assert translator.device == "mps"
