#!/usr/bin/env python3
"""
Example: Custom TTS voice styles.

Run with:
    uv run python examples/custom_voices.py
"""

from qwen3_tts_translator.tts import PortugueseTTS


def main():
    # Initialize TTS with different styles
    styles = [
        ("Ryan", "Speak in a very happy and enthusiastic tone"),
        ("Aiden", "Speak slowly and clearly"),
        ("Ryan", "Speak in a professional business tone"),
    ]

    text = "Olá! Como está você hoje?"

    print("🗣️  TTS Voice Style Examples")
    print("=" * 60)

    for speaker, instruction in styles:
        print(f"\nSpeaker: {speaker}")
        print(f"Style: {instruction}")

        tts = PortugueseTTS(speaker=speaker, instruction=instruction)
        output_file = f"output_{speaker}_{len(instruction)}.wav"

        tts.synthesize_to_file(
            text=text,
            output_path=output_file,
            language="Portuguese",
        )

        print(f"✓ Saved: {output_file}")


if __name__ == "__main__":
    main()
