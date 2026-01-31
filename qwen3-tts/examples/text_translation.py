#!/usr/bin/env python3
"""
Example: Text-only translation (no audio).

Run with:
    uv run python examples/text_translation.py
"""

from qwen3_tts_translator import Qwen3Translator


def main():
    # Initialize translator
    translator = Qwen3Translator()

    # Translate text
    finnish_texts = [
        "Hyvää huomenta",
        "Minun nimeni on Alice",
        "Kuinka voit?",
        "Kiitos avusta",
    ]

    print("🇫🇮 → 🇵🇹 Text Translation Examples")
    print("=" * 60)

    for finnish in finnish_texts:
        portuguese = translator.translate_text(finnish, verbose=False)
        print(f"FI: {finnish}")
        print(f"PT: {portuguese}")
        print()


if __name__ == "__main__":
    main()
