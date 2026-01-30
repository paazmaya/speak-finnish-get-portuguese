#!/usr/bin/env python3
"""
Command-line interface for seamless speech translation.
"""

import argparse
import sys
from pathlib import Path
from .translator import SeamlessTranslator


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Finnish to Portuguese speech translation using SeamlessM4T v2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - Finnish to Portuguese
  seamless-translate input.wav output.wav

  # Specify languages explicitly  
  seamless-translate input.wav output.wav --src-lang fin --tgt-lang por

  # Use CPU (if no GPU available)
  seamless-translate input.wav output.wav --device cpu
  
  # Translate English to Spanish
  seamless-translate input.wav output.wav --src-lang eng --tgt-lang spa

Supported languages: Finnish (fin), Portuguese (por), English (eng),
Spanish (spa), French (fra), German (deu), Italian (ita), Japanese (jpn),
Korean (kor), Chinese (zho), Russian (rus), Arabic (ara), Hindi (hin),
and 90+ more.
        """,
    )

    parser.add_argument(
        "input",
        type=str,
        help="Input audio file path (WAV, MP3, FLAC, etc.)",
    )
    parser.add_argument(
        "output",
        type=str,
        help="Output audio file path (will be saved as WAV)",
    )
    parser.add_argument(
        "--src-lang",
        type=str,
        default="fin",
        help="Source language code (default: fin for Finnish)",
    )
    parser.add_argument(
        "--tgt-lang",
        type=str,
        default="por",
        help="Target language code (default: por for Portuguese)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu/mps). Auto-detected if not specified.",
    )
    parser.add_argument(
        "--8bit",
        action="store_true",
        dest="eight_bit",
        help="Enable 8-bit quantization (saves memory, may reduce quality)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/seamless-m4t-v2-large",
        help="Model to use (default: facebook/seamless-m4t-v2-large)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages",
    )
    parser.add_argument(
        "--list-languages",
        action="store_true",
        help="List supported languages and exit",
    )

    args = parser.parse_args()

    # Handle --list-languages
    if args.list_languages:
        langs = SeamlessTranslator.get_supported_languages()
        print("\nSupported Languages:")
        print("=" * 50)
        for code, name in sorted(langs["common_codes"].items()):
            print(f"  {code:5s} - {name}")
        print("\n" + langs["note"])
        print(f"\nFull list: {langs['docs']}")
        return 0

    # Validate input file exists
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        return 1

    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Initialize translator
        translator = SeamlessTranslator(
            model_name=args.model,
            device=args.device,
            load_in_8bit=args.eight_bit,
        )

        # Perform translation
        translator.translate(
            input_audio_path=args.input,
            output_audio_path=args.output,
            src_lang=args.src_lang,
            tgt_lang=args.tgt_lang,
            verbose=not args.quiet,
        )

        return 0

    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if not args.quiet:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
