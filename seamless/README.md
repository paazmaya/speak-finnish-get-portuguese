# Seamless Speech Translator

Single-model speech-to-speech translation using Meta's SeamlessM4T v2 (8-bit quantized).

## Features

- ✅ **Direct speech-to-speech** translation (no intermediate text needed)
- ✅ **100+ languages** supported
- ✅ **8-bit quantized** for efficient memory usage (~2GB model)
- ✅ **High quality** end-to-end trained model
- ✅ **Fast inference** with GPU acceleration
- ✅ **Simple CLI** and Python API

## Quick Start

### Installation

```bash
# Install using uv
uv pip install -e .
```

### Basic Usage

#### Interactive Mode (Microphone)

```bash
# Start interactive mode with microphone
uv run seamless-interactive

# With specific languages
uv run seamless-interactive --src-lang fin --tgt-lang por

# Use CPU instead of GPU
uv run seamless-interactive --device cpu

# Use a different model
uv run seamless-interactive --model facebook/seamless-m4t-v2-medium
```

**Controls:**
- **SPACE** - Start/stop recording and translate (or interrupt playback)
- **ESC** - Exit

**Note:** Interactive mode shows verbose output by default with detailed timing and processing information.

#### File Translation

```bash
# Basic translation
uv run seamless-translate input.wav output.wav

# With specific languages
uv run seamless-translate input.wav output.wav --src-lang fin --tgt-lang por

# Quiet mode (disable verbose output)
uv run seamless-translate input.wav output.wav --quiet

# Use CPU
uv run seamless-translate input.wav output.wav --device cpu

# List all supported languages
uv run seamless-translate --list-languages
```

**Note:** File translation shows verbose output by default. Use `--quiet` to suppress progress messages and timing information.

```python
from seamless_translator import SeamlessTranslator

# Initialize translator
translator = SeamlessTranslator(
    device="cuda",  # or "cpu" or "mps"
    load_in_8bit=True
)

# Translate
translator.translate(
    input_audio_path="finnish.wav",
    output_audio_path="portuguese.wav",
    src_lang="fin",
    tgt_lang="por"
)
```

#### Interactive Mode (Programmatic)

```python
from seamless_translator import SeamlessTranslator
from seamless_translator.interactive import InteractiveRecorder

# Initialize translator
translator = SeamlessTranslator()

# Start interactive recording
recorder = InteractiveRecorder(
    translator=translator,
    src_lang="fin",
    tgt_lang="por"
)
recorder.run(```

### Python API

```python
from seamless_translator import SeamlessTranslator

# Initialize translator
translator = SeamlessTranslator(
    device="cuda",  # or "cpu" or "mps"
    load_in_8bit=True
)

# Translate
translator.translate(
    input_audio_path="finnish.wav",
    output_audio_path="portuguese.wav",
    src_lang="fin",
    tgt_lang="por"
)
```

## Supported Languages

Common language codes:
- `fin` - Finnish
- `por` - Portuguese
- `eng` - English
- `spa` - Spanish
- `fra` - French
- `deu` - German
- `ita` - Italian
- `jpn` - Japanese
- `kor` - Korean
- `zho` - Chinese
- `rus` - Russian
- `ara` - Arabic
- `hin` - Hindi

And 90+ more languages. Use `seamless-translate --list-languages` for the full list.

## Model Information

- **Model**: [xun/seamless-m4t-v2-large-8bit-bnb](https://huggingface.co/xun/seamless-m4t-v2-large-8bit-bnb)
- **Base**: Meta's SeamlessM4T v2 Large
- **Size**: ~2GB (8-bit quantized)
- **Format**: Safetensors
- **Quality**: State-of-the-art speech-to-speech translation

## Requirements

- Python 3.10+
- CUDA-capable GPU recommended (but works on CPU/MPS)
- ~4GB VRAM (GPU) or ~8GB RAM (CPU)

## Development

### Setup

```bash
# Install with test dependencies
uv pip install -e ".[test]"
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=seamless_translator

# Run specific test
uv run pytest tests/test_translation.py::test_terve_translation -v
```

**Creating Test Audio:**

See [tests/README.md](tests/README.md) for details on creating test fixtures.

Quick start:
```bash
# Record yourself saying "Terve" and save to tests/fixtures/terve.wav
# OR create synthetic test audio:
cd tests/fixtures
python create_test_audio.py
```

### Code Quality

```bash
# Format code
uv run ruff format

# Lint code  
uv run ruff check

# Type checking (if you add type hints)
mypy src/
```

## Integration with Rust Pipeline

You can call this from the main Rust application:

```rust
use std::process::Command;

let status = Command::new("seamless-translate")
    .arg("input.wav")
    .arg("output.wav")
    .arg("--src-lang").arg("fin")
    .arg("--tgt-lang").arg("por")
    .status()?;
```

## Troubleshooting

### Understanding Performance

Both tools show detailed timing information by default:
- Model loading time and size
- Audio loading and preprocessing time
- Translation generation time (usually the slowest part)
- Audio saving time
- Total processing time

Use `--quiet` flag with `seamless-translate` to hide this information.

### Out of Memory
- Use `--device cpu` for CPU inference
- Close other applications
- Use `--no-8bit` flag (uses more memory but may be more stable)

### Slow Performance
- Ensure you're using GPU: `seamless-translate input.wav output.wav --device cuda`
- First run downloads the model (~2GB) - subsequent runs are faster
- CPU inference is expected to be slower

### Model Download Issues
- Model is cached in `~/.cache/huggingface/`
- Ensure you have ~2GB free disk space
- Check internet connection for first download

## License

Uses Meta's SeamlessM4T v2 model (check model card for license terms).

## Links

- [SeamlessM4T v2 Model Card](https://huggingface.co/facebook/seamless-m4t-v2-large)
- [8-bit Quantized Version](https://huggingface.co/xun/seamless-m4t-v2-large-8bit-bnb)
- [Meta's Seamless Communication](https://github.com/facebookresearch/seamless_communication)
