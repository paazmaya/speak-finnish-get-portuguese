"""
Seamless speech-to-speech translation using Meta's SeamlessM4T v2.

This package provides a simple interface for Finnish to Portuguese
(and 100+ other language pairs) speech translation.
"""

__version__ = "0.1.0"

from .translator import SeamlessTranslator

__all__ = ["SeamlessTranslator"]
