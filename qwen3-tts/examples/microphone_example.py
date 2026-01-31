#!/usr/bin/env python3
"""
Example: Microphone recording and translation.

Run with:
    uv run python examples/microphone_example.py
"""

import sys
from qwen3_tts_translator import Qwen3Translator
from qwen3_tts_translator.audio import record_audio, save_audio


def main():
    print("🎙️  Microphone Translation Example")
    print("=" * 50)

    # Initialize translator
    translator = Qwen3Translator(tts_speaker="Aiden")

    # Record 5 seconds of audio
    print("Recording in 3... 2... 1...")
    audio = record_audio(duration=5.0, sample_rate=16000)

    # Save input
    input_path = "recorded_finnish.wav"
    save_audio(audio, input_path, sample_rate=16000)

    # Translate
    output_path = "translated_portuguese.wav"
    translator.translate_audio(
        input_path=input_path,
        output_path=output_path,
        verbose=True,
    )

    print(f"\n✅ Saved to: {output_path}")


if __name__ == "__main__":
    main()
