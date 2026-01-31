#!/bin/bash
# Quick Start Guide for Qwen3-TTS Finnish to Portuguese Translator

set -e

echo "================================================"
echo "Qwen3-TTS Finnish → Portuguese Translator"
echo "================================================"
echo ""

# Check if in correct directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: Run this script from the qwen3-tts directory"
    exit 1
fi

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ Error: uv is not installed"
    echo "   Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi
echo "✓ uv is installed"

# Sync dependencies
echo ""
echo "📦 Installing dependencies..."
uv sync

# Download models
echo ""
echo "⬇️  Downloading models..."
echo "   This will download:"
echo "   - Finnish Whisper STT (~150MB)"
echo "   - Translation models (~300MB)"
echo "   - Qwen3-TTS (~600MB)"
echo "   Total: ~1GB"
echo ""

read -p "Continue? (Y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]] && [[ ! -z $REPLY ]]; then
    echo "⏭️  Skipping model download"
    echo "   Run manually with: uv run qwen-translate --download-models"
else
    uv run qwen-translate --download-models
fi

# Test installation
echo ""
echo "🧪 Testing installation..."
uv run python test_integration.py

echo ""
echo "================================================"
echo "✅ Setup Complete!"
echo "================================================"
echo ""
echo "Quick Start:"
echo ""
echo "  # Interactive mode (recommended)"
echo "  uv run qwen-translate"
echo ""
echo "  # Translate file"
echo "  uv run qwen-translate input.wav output.wav"
echo ""
echo "  # List available voices"
echo "  uv run qwen-translate --list-speakers"
echo ""
echo "Controls in interactive mode:"
echo "  SPACE - Start/stop recording"
echo "  ESC   - Quit"
echo ""
