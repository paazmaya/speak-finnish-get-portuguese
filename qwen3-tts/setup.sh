#!/bin/bash
# Setup script for qwen3-tts-translator

set -e

echo "🚀 Setting up Qwen3-TTS Translator..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ Error: uv is not installed"
    echo "Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    uv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
uv pip install -e ".[test]"

# Download models (if needed)
echo "🤖 Models will be downloaded on first use from HuggingFace"
echo ""
echo "✅ Setup complete!"
echo ""
echo "To activate the environment, run:"
echo "  source .venv/bin/activate"
echo ""
echo "Then try:"
echo "  uv run qwen-translate --download-models  # Download models first"
echo "  uv run qwen-translate                    # Start interactive mode (default)"
echo "  uv run qwen-translate --help             # CLI help"
