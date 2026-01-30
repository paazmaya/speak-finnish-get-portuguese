#!/usr/bin/env python3
"""
Script to create a synthetic 'Terve' audio file for testing.

Since we can't easily generate real speech, this creates a simple test WAV file.
For real testing, record an actual 'Terve' audio sample and save it as terve.wav.
"""

import numpy as np
import soundfile as sf
from pathlib import Path


def create_synthetic_audio(output_path: Path, duration: float = 0.5):
    """
    Create a synthetic audio file (sine wave sweep).
    
    Note: This is NOT real speech - just a placeholder for testing infrastructure.
    Replace with actual recording of someone saying 'Terve' for real tests.
    """
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Create a simple frequency sweep (like a simple tone)
    # This simulates audio data but is NOT actual speech
    frequency_start = 200  # Hz
    frequency_end = 800    # Hz
    frequency = np.linspace(frequency_start, frequency_end, len(t))
    
    # Generate audio signal
    audio = 0.3 * np.sin(2 * np.pi * frequency * t)
    
    # Add some amplitude envelope to make it sound more natural
    envelope = np.exp(-3 * t / duration)
    audio = audio * envelope
    
    # Save as mono WAV
    sf.write(output_path, audio, sample_rate)
    print(f"Created synthetic test audio: {output_path}")
    print(f"Duration: {duration}s, Sample rate: {sample_rate} Hz")
    print("\n⚠️  NOTE: This is synthetic audio, not real speech!")
    print("For proper testing, replace with actual recording of 'Terve'")


def main():
    """Create test audio file."""
    output_path = Path(__file__).parent / "terve.wav"
    
    if output_path.exists():
        response = input(f"{output_path} already exists. Overwrite? [y/N] ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    
    create_synthetic_audio(output_path)
    print(f"\n✓ Test file created: {output_path}")
    print("\nTo record real audio on macOS:")
    print("  1. Use QuickTime Player -> File -> New Audio Recording")
    print("  2. Say 'Terve' (Finnish for Hello)")
    print(f"  3. Save as {output_path}")


if __name__ == "__main__":
    main()
