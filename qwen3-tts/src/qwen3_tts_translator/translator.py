"""
Main translator orchestrating the full pipeline.
"""

import sys
from pathlib import Path
from typing import Optional

from .stt import FinnishSTT
from .translation import TextTranslator
from .tts import PortugueseTTS
from .audio import load_audio, save_audio, resample_audio


class Qwen3Translator:
    """
    Complete Finnish-to-Portuguese speech translation pipeline.

    Stages:
    1. Finnish Speech → Finnish Text (Whisper)
    2. Finnish Text → English Text (Marian MT)
    3. English Text → Portuguese Text (Marian MT)
    4. Portuguese Text → Portuguese Speech (Qwen3-TTS)
    """

    def __init__(
        self,
        device: Optional[str] = None,
        tts_speaker: str = "Ryan",
        tts_instruction: str = "Speak in a clear and natural tone",
    ):
        """
        Initialize the translation pipeline.

        Args:
            device: Device for models (cuda/cpu/mps)
            tts_speaker: Qwen3-TTS speaker voice
            tts_instruction: Voice style instruction
        """
        print("🚀 Initializing Qwen3 Translator...", file=sys.stderr)

        self.stt = FinnishSTT(device=device)
        self.translator = TextTranslator(device=device)
        self.tts = PortugueseTTS(
            device=device, speaker=tts_speaker, instruction=tts_instruction
        )

        print("✓ Pipeline ready!", file=sys.stderr)

    def translate_audio(
        self,
        input_path: str,
        output_path: str,
        verbose: bool = False,
    ):
        """
        Translate audio file from Finnish to Portuguese.

        Args:
            input_path: Input audio file (Finnish speech)
            output_path: Output audio file (Portuguese speech)
            verbose: Print intermediate results
        """
        if verbose:
            print(f"\n📂 Input: {input_path}", file=sys.stderr)
            print(f"📂 Output: {output_path}", file=sys.stderr)

        # Load audio
        audio, sr = load_audio(input_path)

        if verbose:
            print(f"🎵 Loaded audio: {len(audio)/sr:.2f}s @ {sr}Hz", file=sys.stderr)

        # Resample to 16kHz if needed (Whisper requirement)
        if sr != 16000:
            if verbose:
                print("🔄 Resampling to 16kHz...", file=sys.stderr)
            audio = resample_audio(audio, sr, 16000)

        # Stage 1: Speech-to-Text
        if verbose:
            print("\n🎤 Transcribing Finnish speech...", file=sys.stderr)
        finnish_text = self.stt.transcribe(audio, sample_rate=16000)

        if verbose:
            print(f"🇫🇮 Transcribed: {finnish_text}", file=sys.stderr)

        # Stage 2 & 3: Translation
        if verbose:
            print("\n🔄 Translating to Portuguese...", file=sys.stderr)
        portuguese_text = self.translator.translate(finnish_text, verbose=verbose)

        # Stage 4: Text-to-Speech
        if verbose:
            print("\n🗣️  Generating Portuguese speech...", file=sys.stderr)
        self.tts.synthesize_to_file(
            text=portuguese_text,
            output_path=output_path,
            language="Portuguese",
        )

        if verbose:
            print(f"\n✅ Translation complete: {output_path}", file=sys.stderr)

    def translate_text(self, finnish_text: str, verbose: bool = False) -> str:
        """
        Translate Finnish text to Portuguese.

        Args:
            finnish_text: Finnish input text
            verbose: Print intermediate results

        Returns:
            Portuguese translation
        """
        return self.translator.translate(finnish_text, verbose=verbose)

    def speak_portuguese(
        self, text: str, output_path: str, speaker: Optional[str] = None
    ):
        """
        Generate Portuguese speech from text.

        Args:
            text: Portuguese text
            output_path: Output audio file path
            speaker: Optional speaker override
        """
        self.tts.synthesize_to_file(text, output_path, speaker=speaker)
