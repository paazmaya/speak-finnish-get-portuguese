#!/usr/bin/env python3
"""
Example: Download models before using the translator.

Run with:
    uv run python examples/download_models.py
"""

from qwen3_tts_translator.models import download_all_models, list_models


def main():
    print("📦 Model Download Example")
    print("=" * 60)

    # Show current model status
    print("\n1️⃣  Current model status:")
    list_models()

    # Download all models
    print("\n2️⃣  Downloading all models...")
    try:
        download_all_models()
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

    # Show updated status
    print("\n3️⃣  Updated model status:")
    list_models()

    print("\n✅ All models are ready to use!")
    print("\nModels are stored in: ./models/")
    print("  - Finnish-NLP--whisper-tiny-finnish/")
    print("  - Helsinki-NLP--opus-tatoeba-fi-en/")
    print("  - Helsinki-NLP--opus-mt-tc-big-en-pt/")
    print("  - Qwen--Qwen3-TTS-12Hz-0.6B-CustomVoice/")

    return 0


if __name__ == "__main__":
    exit(main())
