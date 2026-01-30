#!/usr/bin/env python3
"""
Record a test audio file using your microphone.

This script helps you record yourself saying "Terve" for testing.
"""

import sounddevice as sd
import soundfile as sf
import numpy as np
from pathlib import Path
import sys


def record_test_audio(output_path: Path, duration: float = 2.0, sample_rate: int = 16000):
    """
    Record audio from microphone.
    
    Args:
        output_path: Where to save the audio
        duration: Recording duration in seconds
        sample_rate: Audio sample rate (16kHz for SeamlessM4T)
    """
    print(f"\n🎤 Recording for {duration} seconds...")
    print("Say 'Terve' clearly!\n")
    
    # Countdown
    for i in range(3, 0, -1):
        print(f"Starting in {i}...")
        sd.sleep(1000)
    
    print("🔴 RECORDING NOW!")
    
    # Record
    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype='float32'
    )
    sd.wait()  # Wait for recording to finish
    
    print("✓ Recording complete!")
    
    # Save to file
    sf.write(output_path, audio, sample_rate)
    print(f"✓ Saved to: {output_path}")
    
    # Play back
    print("\n🔊 Playing back...")
    sd.play(audio, sample_rate)
    sd.wait()
    
    print("\n✓ Done!")


def main():
    """Main entry point."""
    output_path = Path(__file__).parent / "terve.wav"
    
    print("=" * 60)
    print("  Test Audio Recorder - 'Terve' (Finnish for Hello)")
    print("=" * 60)
    
    if output_path.exists():
        print(f"\n⚠️  File already exists: {output_path}")
        response = input("Overwrite? [y/N] ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    
    print("\nInstructions:")
    print("  1. Ensure your microphone is working")
    print("  2. When recording starts, say 'Terve' clearly")
    print("  3. The recording will play back for verification")
    print()
    
    response = input("Ready to record? [y/N] ")
    if response.lower() != 'y':
        print("Aborted.")
        return
    
    try:
        record_test_audio(output_path)
        
        print("\n" + "=" * 60)
        print("You can now run tests with:")
        print("  cd ../..")
        print("  uv run pytest tests/test_translation.py::test_terve_translation -v")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\nRecording cancelled.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
