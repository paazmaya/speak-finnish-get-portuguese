"""
Test configuration and fixtures.
"""

import pytest


@pytest.fixture(scope="session")
def sample_rate():
    """Standard sample rate for testing."""
    return 16000


@pytest.fixture(scope="session")
def test_text_finnish():
    """Sample Finnish text for testing."""
    return "Hyvää huomenta"


@pytest.fixture(scope="session")
def test_text_english():
    """Sample English text for testing."""
    return "Good morning"


@pytest.fixture(scope="session")
def test_text_portuguese():
    """Sample Portuguese text for testing."""
    return "Bom dia"
