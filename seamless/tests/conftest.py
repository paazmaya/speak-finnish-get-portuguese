"""Pytest configuration and fixtures."""

import pytest
from pathlib import Path


@pytest.fixture
def test_fixtures_dir():
    """Return path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def terve_audio_path(test_fixtures_dir):
    """Return path to the 'terve' test audio file."""
    return test_fixtures_dir / "terve.wav"
