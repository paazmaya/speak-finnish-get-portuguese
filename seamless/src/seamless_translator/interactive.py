"""
Interactive microphone recording and translation with keyboard controls.
"""

import sys
import threading
import queue
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd
import soundfile as sf
from pynput import keyboard

from .translator import SeamlessTranslator


class InteractiveRecorder:
    """
    Interactive recorder with keyboard controls.

    Press SPACE to start/stop recording and trigger translation.
    Press ESC to quit.
    """

    def __init__(
        self,
        translator: SeamlessTranslator,
        src_lang: str = "fin",
        tgt_lang: str = "por",
        sample_rate: int = 16000,
    ):
        """
        Initialize interactive recorder.

        Args:
            translator: SeamlessTranslator instance
            src_lang: Source language code
            tgt_lang: Target language code
            sample_rate: Audio sample rate (16kHz for SeamlessM4T)
        """
        self.translator = translator
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.sample_rate = sample_rate

        self.is_recording = False
        self.is_playing = False
        self.should_stop_playback = False
        self.should_exit = False

        self.audio_queue = queue.Queue()
        self.recorded_frames = []

        self.recording_stream: Optional[sd.InputStream] = None
        self.playback_stream: Optional[sd.OutputStream] = None

    def audio_callback(self, indata, frames, time_info, status):
        """Callback for audio recording."""
        if status:
            print(f"Recording status: {status}", file=sys.stderr)

        if self.is_recording:
            # Copy audio data to avoid issues with buffer reuse
            self.recorded_frames.append(indata.copy())

    def start_recording(self):
        """Start recording audio."""
        if self.is_recording:
            return

        print("\n🎤 Recording... (Press SPACE to stop)", file=sys.stderr)
        self.is_recording = True
        self.recorded_frames = []

        # Start audio stream
        self.recording_stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            callback=self.audio_callback,
        )
        self.recording_stream.start()

    def stop_recording(self):
        """Stop recording and process audio."""
        if not self.is_recording:
            return

        self.is_recording = False

        # Stop stream
        if self.recording_stream:
            self.recording_stream.stop()
            self.recording_stream.close()
            self.recording_stream = None

        # Check if we have audio
        if not self.recorded_frames:
            print("⚠️  No audio recorded", file=sys.stderr)
            return

        # Combine recorded frames
        audio_data = np.concatenate(self.recorded_frames, axis=0)
        duration = len(audio_data) / self.sample_rate

        print(f"✓ Recorded {duration:.2f} seconds", file=sys.stderr)

        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_input:
            sf.write(tmp_input.name, audio_data, self.sample_rate)
            input_path = tmp_input.name

        # Translate
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_output:
            output_path = tmp_output.name

        try:
            print(
                f"🔄 Translating {self.src_lang} → {self.tgt_lang}...", file=sys.stderr
            )
            print(f"📁 Input audio: {input_path}", file=sys.stderr)
            print(f"📁 Output audio: {output_path}", file=sys.stderr)

            self.translator.translate(
                input_audio_path=input_path,
                output_audio_path=output_path,
                src_lang=self.src_lang,
                tgt_lang=self.tgt_lang,
                verbose=True,
            )

            print("✓ Translation complete!", file=sys.stderr)

            # Play translation
            self.play_audio(output_path)

        finally:
            # Cleanup temp files
            Path(input_path).unlink(missing_ok=True)
            Path(output_path).unlink(missing_ok=True)

    def play_audio(self, audio_path: str):
        """Play audio file with interruption support."""
        print(
            "\n🔊 Playing translation... (Press SPACE to interrupt)", file=sys.stderr
        )
        print(f"📁 Playing file: {audio_path}", file=sys.stderr)

        self.is_playing = True
        self.should_stop_playback = False

        # Load audio
        data, sample_rate = sf.read(audio_path, dtype="float32")

        print(
            f"🎵 Playback started ({len(data) / sample_rate:.2f}s at {sample_rate} Hz)",
            file=sys.stderr,
        )

        # Ensure mono
        if len(data.shape) > 1:
            data = data.mean(axis=1)

        # Play in chunks to allow interruption
        position = 0

        def audio_callback(outdata, frames, time_info, status):
            nonlocal position

            if status:
                print(f"Playback status: {status}", file=sys.stderr)

            if self.should_stop_playback or position >= len(data):
                # Fill with silence
                outdata[:] = 0
                raise sd.CallbackStop

            # Get chunk
            end = min(position + frames, len(data))
            chunk = data[position:end]

            # Pad if necessary
            if len(chunk) < frames:
                chunk = np.pad(chunk, (0, frames - len(chunk)))

            outdata[:, 0] = chunk
            position = end

        try:
            with sd.OutputStream(
                samplerate=sample_rate,
                channels=1,
                dtype="float32",
                callback=audio_callback,
            ):
                # Wait until playback is finished or interrupted
                while position < len(data) and not self.should_stop_playback:
                    sd.sleep(100)

        except Exception as e:
            print(f"Playback error: {e}", file=sys.stderr)

        finally:
            self.is_playing = False

            if self.should_stop_playback:
                print("⏸️  Playback interrupted", file=sys.stderr)
            else:
                print("✓ Playback finished", file=sys.stderr)

    def on_press(self, key):
        """Handle key press events."""
        try:
            # Handle space key
            if key == keyboard.Key.space:
                if self.is_playing:
                    # Interrupt playback and start recording
                    self.should_stop_playback = True
                    # Wait a bit for playback to stop
                    threading.Timer(0.2, self.start_recording).start()
                elif self.is_recording:
                    # Stop recording and translate
                    self.stop_recording()
                else:
                    # Start recording
                    self.start_recording()

                return False  # Stop listener temporarily to prevent key repeat

            # Handle ESC key
            elif key == keyboard.Key.esc:
                print("\n👋 Exiting...", file=sys.stderr)
                self.should_exit = True

                # Stop any ongoing recording
                if self.is_recording:
                    self.is_recording = False
                    if self.recording_stream:
                        self.recording_stream.stop()
                        self.recording_stream.close()

                # Stop any ongoing playback
                if self.is_playing:
                    self.should_stop_playback = True

                return False  # Stop listener

        except AttributeError:
            # Handle other keys (non-special keys)
            pass

    def on_release(self, key):
        """Handle key release events."""
        # Restart listener after space key is released
        if key == keyboard.Key.space:
            return False

    def run(self):
        """Run the interactive recorder."""
        print("\n" + "=" * 60)
        print("  🎙️  Interactive Speech Translator")
        print("=" * 60)
        print(f"\nLanguages: {self.src_lang} → {self.tgt_lang}")
        print("\nControls:")
        print("  SPACE - Start/stop recording and translate")
        print("           (or interrupt playback to start recording)")
        print("  ESC   - Exit")
        print("\n" + "=" * 60)
        print("\n⏸️  Waiting... Press SPACE to start recording.\n")

        # Start keyboard listener
        while not self.should_exit:
            with keyboard.Listener(
                on_press=self.on_press, on_release=self.on_release
            ) as listener:
                listener.join()

            # Small delay to prevent CPU spinning
            if not self.should_exit:
                sd.sleep(100)

        print("Goodbye! 👋\n")


def main():
    """Main entry point for interactive mode."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Interactive speech translation with microphone",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Controls:
  SPACE - Start/stop recording and translate
          (or interrupt playback to start recording)
  ESC   - Exit

Example:
  seamless-interactive --src-lang fin --tgt-lang por
        """,
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
        "--model",
        type=str,
        default="facebook/seamless-m4t-v2-large",
        help="Model to use (default: facebook/seamless-m4t-v2-large)",
    )
    parser.add_argument(
        "--8bit",
        action="store_true",
        dest="eight_bit",
        help="Enable 8-bit quantization (saves memory, may reduce quality)",
    )

    args = parser.parse_args()

    try:
        # Initialize translator
        print("\n" + "=" * 60, file=sys.stderr)
        print("🚀 Initializing Seamless Translator", file=sys.stderr)
        print("=" * 60, file=sys.stderr)

        translator = SeamlessTranslator(
            model_name=args.model,
            device=args.device,
            load_in_8bit=args.eight_bit,
        )

        print("=" * 60, file=sys.stderr)
        print("✓ Initialization complete! Ready to translate.", file=sys.stderr)
        print("=" * 60 + "\n", file=sys.stderr)

        # Start interactive mode
        recorder = InteractiveRecorder(
            translator=translator,
            src_lang=args.src_lang,
            tgt_lang=args.tgt_lang,
        )
        recorder.run()

        return 0

    except KeyboardInterrupt:
        print("\n\nInterrupted by user", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
