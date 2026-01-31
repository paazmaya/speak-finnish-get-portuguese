#!/usr/bin/env python3
"""
Example: Basic translation from audio file.

Run with:
    uv run python examples/basic_translation.py
"""

from qwen3_tts_translator import Qwen3Translator


def main():
    # Initialize translator (auto-detects GPU)
    translator = Qwen3Translator(
        tts_speaker="Ryan",
        tts_instruction="Speak in a clear and friendly tone",
    )

    # Translate audio file
    translator.translate_audio(
        input_path="input_finnish.wav",
        output_path="output_portuguese.wav",
        verbose=True,
    )

    print("✅ Translation complete!")


if __name__ == "__main__":
    main()
