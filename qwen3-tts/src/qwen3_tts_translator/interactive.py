#!/usr/bin/env python3
"""
Interactive microphone recording and translation.
"""

import sys
import tempfile
from pathlib import Path
from typing import Optional

import sounddevice as sd
from pynput import keyboard

from .translator import Qwen3Translator
from .audio import AudioRecorder, save_audio


class InteractiveTranslator:
    """
    Interactive Finnish-to-Portuguese translator with microphone.

    Controls:
    - SPACE: Start/stop recording and translate
    - ESC: Quit
    """

    def __init__(
        self,
        translator: Qwen3Translator,
        sample_rate: int = 16000,
    ):
        """
        Initialize interactive translator.

        Args:
            translator: Qwen3Translator instance
            sample_rate: Recording sample rate (16kHz for Whisper)
        """
        self.translator = translator
        self.sample_rate = sample_rate
        self.recorder = AudioRecorder(sample_rate=sample_rate)
        self.should_exit = False
        self.is_playing = False

    def on_press(self, key):
        """Handle key press events."""
        try:
            # ESC to quit
            if key == keyboard.Key.esc:
                print("\n👋 Exiting...", file=sys.stderr)
                self.should_exit = True
                return False

            # SPACE to start/stop recording
            if key == keyboard.Key.space:
                if self.recorder.is_recording:
                    self._stop_and_translate()
                else:
                    self._start_recording()

        except Exception as e:
            print(f"❌ Error: {e}", file=sys.stderr)

    def _start_recording(self):
        """Start recording audio."""
        print("\n🎤 Recording... (Press SPACE to stop)", file=sys.stderr)
        self.recorder.start()

    def _stop_and_translate(self):
        """Stop recording and process translation."""
        audio = self.recorder.stop()

        if len(audio) == 0:
            print("⚠️  No audio recorded", file=sys.stderr)
            return

        duration = len(audio) / self.sample_rate
        print(f"✓ Recorded {duration:.2f}s", file=sys.stderr)

        # Save to temp file
        with tempfile.NamedTemporaryFile(
            suffix=".wav", delete=False
        ) as tmp_input:
            save_audio(audio, tmp_input.name, self.sample_rate)
            input_path = tmp_input.name

        with tempfile.NamedTemporaryFile(
            suffix=".wav", delete=False
        ) as tmp_output:
            output_path = tmp_output.name

        try:
            # Translate
            print("🔄 Translating Finnish → Portuguese...", file=sys.stderr)
            self.translator.translate_audio(
                input_path=input_path,
                output_path=output_path,
                verbose=True,
            )

            # Play result
            self._play_audio(output_path)

        finally:
            # Cleanup
            Path(input_path).unlink(missing_ok=True)
            Path(output_path).unlink(missing_ok=True)

    def _play_audio(self, audio_path: str):
        """Play translated audio."""
        import soundfile as sf

        print("\n🔊 Playing translation...", file=sys.stderr)

        try:
            audio, sr = sf.read(audio_path, dtype="float32")
            sd.play(audio, samplerate=sr)
            sd.wait()
            print("✓ Playback complete", file=sys.stderr)
        except Exception as e:
            print(f"❌ Playback error: {e}", file=sys.stderr)

    def run(self):
        """Run interactive session."""
        print("\n🎙️  Interactive Qwen3-TTS Translator", file=sys.stderr)
        print("=" * 50, file=sys.stderr)
        print("Controls:", file=sys.stderr)
        print("  SPACE - Start/stop recording and translate", file=sys.stderr)
        print("  ESC   - Quit", file=sys.stderr)
        print("\nReady! Press SPACE to start recording...\n", file=sys.stderr)

        # Start keyboard listener
        with keyboard.Listener(on_press=self.on_press) as listener:
            listener.join()


def main():
    """Interactive mode entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Interactive Finnish to Portuguese translation"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda/cpu/mps)",
    )
    parser.add_argument(
        "--speaker",
        type=str,
        default="Ryan",
        help="TTS speaker (default: Ryan)",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default="Speak in a clear and natural tone",
        help="Voice instruction",
    )

    args = parser.parse_args()

    try:
        # Initialize translator
        translator = Qwen3Translator(
            device=args.device,
            tts_speaker=args.speaker,
            tts_instruction=args.instruction,
        )

        # Run interactive session
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


if __name__ == "__main__":
    sys.exit(main())
