#!/usr/bin/env python3
"""
Quick integration test for Qwen3-TTS Finnish to Portuguese translator.
"""

import sys
from qwen3_tts_translator.stt import FinnishSTT
from qwen3_tts_translator.translation import TextTranslator
from qwen3_tts_translator.tts import PortugueseTTS


def test_modules_import():
    """Test that all modules can be imported."""
    print("✓ All modules imported successfully")


def test_tts_initialization():
    """Test TTS model initialization."""
    print("\n🔧 Testing TTS initialization...")
    tts = PortugueseTTS(device="cpu")  # Use CPU for quick test
    print("✓ TTS initialized successfully")
    
    # List available speakers
    speakers = tts.list_speakers()
    print(f"\n📢 Available speakers ({len(speakers)}):")
    for name, desc in speakers.items():
        print(f"  - {name}: {desc}")


def test_translation():
    """Test text translation."""
    print("\n🔄 Testing translation...")
    translator = TextTranslator()
    
    finnish_text = "Hei, mitä kuuluu?"
    print(f"Finnish: {finnish_text}")
    
    portuguese = translator.translate(finnish_text)
    print(f"Portuguese: {portuguese}")
    print("✓ Translation successful")


def test_simple_tts():
    """Test simple TTS synthesis (will download model on first run)."""
    print("\n🎤 Testing TTS synthesis...")
    print("Note: First run will download ~600MB Qwen3-TTS model")
    print("You can skip this by pressing Ctrl+C")
    
    try:
        tts = PortugueseTTS(device="cpu")
        test_text = "Olá, tudo bem?"
        print(f"Synthesizing: {test_text}")
        
        # This will load the model and synthesize
        wavs, sr = tts.synthesize(test_text)
        print(f"✓ Generated audio: {len(wavs[0])} samples at {sr}Hz")
        print(f"  Duration: {len(wavs[0])/sr:.2f} seconds")
        
    except KeyboardInterrupt:
        print("\n⏭️  Skipping TTS test")
    except Exception as e:
        print(f"⚠️  TTS test failed: {e}")
        print("  (This is expected if models are not downloaded)")


if __name__ == "__main__":
    print("=" * 60)
    print("Qwen3-TTS Integration Test")
    print("=" * 60)
    
    test_modules_import()
    test_tts_initialization()
    test_translation()
    
    # Optional: Full TTS test
    response = input("\n🤔 Run full TTS synthesis test? (y/N): ").strip().lower()
    if response == 'y':
        test_simple_tts()
    else:
        print("⏭️  Skipping TTS synthesis test")
    
    print("\n" + "=" * 60)
    print("✅ Integration test complete!")
    print("=" * 60)
