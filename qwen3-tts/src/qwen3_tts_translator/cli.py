#!/usr/bin/env python3
"""
Command-line interface for Qwen3-TTS translation.
"""

import argparse
import sys
from pathlib import Path

from .translator import Qwen3Translator
from .tts import PortugueseTTS
from .models import download_all_models, list_models
from .interactive import InteractiveTranslator


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Finnish to Portuguese speech translation using Qwen3-TTS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (default - no arguments)
  qwen-translate

  # Basic translation from file
  qwen-translate input.wav output.wav

  # Use CPU instead of GPU
  qwen-translate input.wav output.wav --device cpu

  # Use different speaker
  qwen-translate input.wav output.wav --speaker Aiden

  # Custom voice instruction
  qwen-translate input.wav output.wav --instruction "Speak slowly and clearly"

  # Verbose mode
  qwen-translate input.wav output.wav --verbose

  # Download models
  qwen-translate --download-models

Available speakers: Ryan, Aiden (recommended for Portuguese)
        """,
    )

    parser.add_argument(
        "input",
        type=str,
        nargs="?",
        help="Input audio file (Finnish speech)",
    )
    parser.add_argument(
        "output",
        type=str,
        nargs="?",
        help="Output audio file (Portuguese speech)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda/cpu/mps). Auto-detected if not specified.",
    )
    parser.add_argument(
        "--speaker",
        type=str,
        default="Aiden",
        help="TTS speaker voice (default: Aiden)",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default="Speak in a clear and natural tone",
        help="Voice style instruction",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed progress",
    )
    parser.add_argument(
        "--list-speakers",
        action="store_true",
        help="List available TTS speakers",
    )
    parser.add_argument(
        "--download-models",
        action="store_true",
        help="Download all required models",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List models and their download status",
    )

    args = parser.parse_args()

    # Handle --download-models
    if args.download_models:
        try:
            download_all_models()
            return 0
        except Exception as e:
            print(f"❌ Error downloading models: {e}", file=sys.stderr)
            return 1

    # Handle --list-models
    if args.list_models:
        list_models()
        return 0

    # Handle --list-speakers
    if args.list_speakers:
        print("\nAvailable Qwen3-TTS Speakers:")
        print("=" * 60)
        for speaker, desc in PortugueseTTS.list_speakers().items():
            print(f"  {speaker:12s} - {desc}")
        print("\nRecommended for Portuguese: Ryan, Aiden")
        return 0

    # If no input/output provided, launch interactive mode
    if not args.input and not args.output:
        print("🎙️  Starting interactive mode...")
        print("    (Press SPACE to record, SPACE again to translate, ESC to quit)")
        print()
        try:
            translator = Qwen3Translator(
                device=args.device,
                tts_speaker=args.speaker,
                tts_instruction=args.instruction,
            )
            interactive = InteractiveTranslator(translator)
            interactive.run()
            return 0
        except KeyboardInterrupt:
            print("\n👋 Interrupted", file=sys.stderr)
            return 0
        except Exception as e:
            print(f"❌ Error: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            return 1

    # Require both input and output for file translation
    if not args.input or not args.output:
        parser.error("both input and output are required for file translation")

    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Error: Input file not found: {args.input}", file=sys.stderr)
        return 1

    # Initialize translator
    try:
        translator = Qwen3Translator(
            device=args.device,
            tts_speaker=args.speaker,
            tts_instruction=args.instruction,
        )
    except Exception as e:
        print(f"❌ Error initializing translator: {e}", file=sys.stderr)
        return 1

    # Translate
    try:
        translator.translate_audio(
            input_path=str(input_path),
            output_path=args.output,
            verbose=args.verbose,
        )
        return 0
    except Exception as e:
        print(f"❌ Translation failed: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
