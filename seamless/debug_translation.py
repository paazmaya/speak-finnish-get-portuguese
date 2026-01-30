#!/usr/bin/env python3
"""
Debug script to diagnose translation issues.

This script runs a translation and shows detailed diagnostics.
"""

import sys
from pathlib import Path
import soundfile as sf
import tempfile

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from seamless_translator import SeamlessTranslator


def debug_translation(input_path: Path):
    """Run translation with detailed diagnostics."""
    print("=" * 70)
    print("  TRANSLATION DIAGNOSTICS")
    print("=" * 70)
    
    # Check input file
    print(f"\n📁 Input File: {input_path}")
    if not input_path.exists():
        print(f"❌ File does not exist!")
        return False
    
    # Read input audio
    print("\n📊 Input Audio Analysis:")
    try:
        audio_data, sample_rate = sf.read(str(input_path))
        duration = len(audio_data) / sample_rate
        channels = audio_data.shape[1] if len(audio_data.shape) > 1 else 1
        
        print(f"  ✓ File readable")
        print(f"  Duration:    {duration:.3f} seconds")
        print(f"  Sample rate: {sample_rate} Hz")
        print(f"  Channels:    {channels}")
        print(f"  Samples:     {len(audio_data)}")
        print(f"  Data type:   {audio_data.dtype}")
        print(f"  Range:       [{audio_data.min():.3f}, {audio_data.max():.3f}]")
        
        if duration < 0.1:
            print(f"  ⚠️  WARNING: Audio very short (< 0.1s)")
        if duration > 10:
            print(f"  ⚠️  WARNING: Audio very long (> 10s)")
        if abs(audio_data.max()) < 0.01:
            print(f"  ⚠️  WARNING: Audio level very low")
            
    except Exception as e:
        print(f"  ❌ Error reading audio: {e}")
        return False
    
    # Initialize translator
    print("\n🤖 Initializing Translator:")
    try:
        translator = SeamlessTranslator(device="cpu", load_in_8bit=False)
        print(f"  ✓ Translator initialized")
        print(f"  Device: {translator.device}")
        print(f"  Model:  {translator.model_name}")
    except Exception as e:
        print(f"  ❌ Error initializing translator: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Run translation
    print("\n🔄 Running Translation:")
    output_path = Path(tempfile.mktemp(suffix=".wav"))
    
    try:
        translator.translate(
            input_audio_path=str(input_path),
            output_audio_path=str(output_path),
            src_lang="fin",
            tgt_lang="por",
            verbose=True,
        )
        print(f"\n✓ Translation completed!")
    except Exception as e:
        print(f"\n❌ Translation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Analyze output
    print("\n📊 Output Audio Analysis:")
    try:
        if not output_path.exists():
            print(f"  ❌ Output file was not created!")
            return False
        
        output_data, output_sr = sf.read(str(output_path))
        output_duration = len(output_data) / output_sr
        output_channels = output_data.shape[1] if len(output_data.shape) > 1 else 1
        
        print(f"  ✓ Output file created: {output_path}")
        print(f"  Duration:    {output_duration:.3f} seconds")
        print(f"  Sample rate: {output_sr} Hz")
        print(f"  Channels:    {output_channels}")
        print(f"  Samples:     {len(output_data)}")
        print(f"  Range:       [{output_data.min():.3f}, {output_data.max():.3f}]")
        
        if output_duration < 0.1:
            print(f"  ❌ Output audio too short! Translation likely failed.")
            return False
        if abs(output_data.max()) < 0.01:
            print(f"  ⚠️  WARNING: Output audio level very low")
        
        print(f"\n🎵 Playing output audio...")
        import sounddevice as sd
        sd.play(output_data, output_sr)
        sd.wait()
        print(f"✓ Playback complete")
        
        # Cleanup
        output_path.unlink()
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error analyzing output: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python debug_translation.py <audio_file>")
        print("\nExample:")
        print("  python debug_translation.py tests/fixtures/terve.wav")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    success = debug_translation(input_path)
    
    print("\n" + "=" * 70)
    if success:
        print("✓ DIAGNOSIS: Translation pipeline working correctly!")
    else:
        print("❌ DIAGNOSIS: Issues detected - see errors above")
    print("=" * 70)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
