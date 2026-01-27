# Coding Agent Instructions for speech-fin-to-por

## Project Overview

This is a Rust-based speech translation pipeline that translates spoken Finnish to spoken Portuguese using the Candle ML framework. The pipeline consists of 5 stages:

1. **Audio Capture** - Record Finnish speech from microphone or load from WAV file
2. **Speech-to-Text** - Transcribe Finnish audio using Whisper Finnish model
3. **Translation** - Translate Finnish → English → Portuguese using Helsinki-NLP Marian models
4. **Text-to-Speech** - Synthesize Portuguese speech using Parler TTS
5. **Evaluation** (optional) - Validate output quality using Whisper Portuguese model

## Technology Stack

- **Language**: Rust 2021 edition
- **ML Framework**: Candle (Facebook's Rust ML framework)
- **ONNX Runtime**: For TTS model inference
- **Audio**: cpal (microphone), hound (WAV encoding)
- **Models**: HuggingFace models downloaded via hf-hub
- **Supported Devices**: CPU, CUDA, Metal (Apple Silicon)

## Project Structure

### Core Modules

- **[src/main.rs](src/main.rs)** - CLI argument parsing, pipeline orchestration, main execution flow
- **[src/config.rs](src/config.rs)** - Model configuration, file paths, HuggingFace model downloading
- **[src/audio_capture.rs](src/audio_capture.rs)** - Microphone recording, WAV file I/O, audio preprocessing
- **[src/speech_to_text.rs](src/speech_to_text.rs)** - Whisper model inference for Finnish STT
- **[src/translation.rs](src/translation.rs)** - Two-stage Marian MT (Finnish→English→Portuguese)
- **[src/text_to_speech.rs](src/text_to_speech.rs)** - Parler TTS ONNX model for Portuguese synthesis
- **[src/speech_evaluator.rs](src/speech_evaluator.rs)** - Quality validation using Whisper Portuguese

### Key Configuration Files

- **[Cargo.toml](Cargo.toml)** - Rust dependencies and feature flags (cuda, metal, accelerate, mkl)
- **[build.sh](build.sh)** - Build script with environment variable setup (LIBCLANG_PATH, HF_HOME)
- **[models/](models/)** - Directory containing downloaded HuggingFace models (safetensors, ONNX, tokenizers)

### Binary Data

- **[src/melfilters.bytes](src/melfilters.bytes)** - Pre-computed mel filterbank for 80-channel spectrograms
- **[src/melfilters128.bytes](src/melfilters128.bytes)** - Pre-computed mel filterbank for 128-channel spectrograms

## Development Guidelines

### Code Style

1. **Error Handling**: Always use `anyhow::Result<T>` for functions that can fail. Use `.context("...")` to add context to errors.
   ```rust
   fn load_model(path: &Path) -> Result<Model> {
       let config = std::fs::read_to_string(path)
           .context("Failed to read config file")?;
       // ...
   }
   ```

2. **Device Management**: Support CPU, CUDA, and Metal backends. Always use `Device` parameter for model loading:
   ```rust
   fn new(model_path: &Path, device: &Device) -> Result<Self> {
       let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[safetensors_path], DTYPE, device)? };
   }
   ```

3. **Audio Format**: Internal audio format is:
   - Sample rate: 16kHz (Whisper requirement)
   - Channels: Mono (single channel)
   - Format: f32 samples normalized to [-1.0, 1.0]

4. **Model Loading**: Use `hf-hub` for downloading models. Always verify files exist before loading:
   ```rust
   model_config.verify_files(&models_dir)?;
   ```

### Working with Models

#### Adding a New Model

1. Define model configuration in [src/config.rs](src/config.rs):
   ```rust
   pub const NEW_MODEL_ID: &str = "org/model-name";
   pub const NEW_MODEL: ModelConfig = ModelConfig {
       id: NEW_MODEL_ID,
       files: &["config.json", "model.safetensors", "tokenizer.json"],
   };
   ```

2. Add download logic in `download_model_file()` function
3. Implement model loading in appropriate module (e.g., `src/new_model.rs`)
4. Update pipeline in [src/main.rs](src/main.rs)

#### Model File Formats

- **Whisper models**: SafeTensors format with tokenizer JSON
- **Marian MT models**: SafeTensors with SentencePiece tokenizers (.spm files)
- **Parler TTS**: ONNX format with custom tokenizer JSON
- **MADLAD**: GGUF quantized format (currently unused, T5 model)

### Audio Processing

#### Recording from Microphone

```rust
let audio_capture = AudioCapture::new();
let samples = audio_capture.record_from_microphone(duration_secs)?;
```

- Uses cpal to access default microphone
- Automatically resamples to 16kHz
- Converts stereo to mono if needed
- Normalizes audio to [-1.0, 1.0]

#### Segmented Recording

Enable with `--segmented` flag:
- Press SPACE to save current segment
- Press ENTER to finish and process all segments
- Each segment saved to separate file
- All segments concatenated for translation

### Translation Pipeline

The translation uses a two-stage approach:

1. **Finnish → English**: `Helsinki-NLP/opus-tatoeba-fi-en`
2. **English → Portuguese**: `Helsinki-NLP/opus-mt-tc-big-en-pt`

This provides better quality than direct Finnish→Portuguese due to:
- Limited training data for Finnish-Portuguese pairs
- Better coverage via high-resource English intermediate

### Building and Running

#### Prerequisites (macOS)

```bash
brew install cmake llvm
export LIBCLANG_PATH="/opt/homebrew/opt/llvm/lib"
```

#### Build Commands

```bash
# CPU-only build
./build.sh

# With Metal (Apple Silicon)
./build.sh --features metal

# With CUDA
./build.sh --features cuda
```

#### Common Run Commands

```bash
# Record 5 seconds, translate, and speak
cargo run --release -- --microphone --duration 5

# Segmented recording with space key
cargo run --release -- --segmented

# Test mode with custom text
cargo run --release -- --test-mode --test-text "Hyvää huomenta"

# Evaluate output quality
cargo run --release -- --test-mode --evaluate

# Process existing audio file
cargo run --release -- --input input.wav --output output.wav
```

### Testing

#### Test Mode

Use `--test-mode` to skip audio capture and test translation pipeline:
```bash
cargo run --release -- --test-mode --test-text "Minun nimeni on Alice"
```

#### Evaluation Mode

Use `--evaluate` (test mode only) to validate TTS output:
- Transcribes generated Portuguese audio back to text
- Compares with expected translation
- Reports word accuracy and missing words

### Troubleshooting

#### Common Issues

1. **LIBCLANG_PATH error**: Set environment variable before building
   ```bash
   export LIBCLANG_PATH="/opt/homebrew/opt/llvm/lib"
   ```

2. **Missing models**: Run model download command
   ```bash
   cargo run --release -- --download-models
   ```

3. **Audio device errors**: Check microphone permissions in System Settings

4. **Out of memory**: Use smaller models or run on CPU with `--cpu` flag

5. **Slow inference**: Enable hardware acceleration:
   - macOS: `--features metal`
   - Linux/Windows with NVIDIA: `--features cuda`

### Performance Optimization

1. **Use Release Build**: Always use `--release` for inference (10-100x faster)
2. **Enable Hardware Acceleration**: Use Metal/CUDA features
3. **Model Size**: Consider model size vs. quality tradeoff:
   - tiny: Fast, lower quality
   - small: Balanced
   - base/large: Slower, higher quality

### Code Patterns

#### Loading Safetensors Models

```rust
let safetensors_path = model_path.join("model.safetensors");
let vb = unsafe {
    VarBuilder::from_mmaped_safetensors(&[safetensors_path], DTYPE, device)?
};
let model = Model::load(vb, &config)?;
```

#### Tokenization

```rust
let tokenizer = Tokenizer::from_file(tokenizer_path)
    .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
let tokens = tokenizer.encode(text, true)
    .map_err(|e| anyhow::anyhow!("Encoding failed: {}", e))?;
```

#### Mel Spectrogram Generation

```rust
let mel_filters = include_bytes!("melfilters.bytes");
let mel = audio_to_mel(&samples, mel_filters)?;
let mel_tensor = Tensor::from_vec(mel, (1, 80, mel_len), device)?;
```

## Dependencies and Versioning

- **candle-core/nn/transformers**: 0.9.2-alpha.2 (bleeding edge)
- **ort (ONNX Runtime)**: 2.0.0-rc.11 with download-binaries feature
- **tokenizers**: 0.21 (HuggingFace tokenizers)
- **hound**: 3.5 (WAV I/O)
- **cpal**: 0.15 (audio capture)

## Contributing Guidelines

1. **Test Changes**: Always test with both `--test-mode` and `--microphone` modes
2. **Error Messages**: Provide helpful error messages with context
3. **Documentation**: Update README.md when changing pipeline or adding features
4. **Cross-platform**: Consider macOS, Linux, and Windows compatibility
5. **Model Files**: Don't commit large model files - use `--download-models` approach

## Important Notes

- Models are stored in `./models/` directory (git-ignored)
- Model subdirectories use `--` separator (e.g., `Helsinki-NLP--opus-mt-fi-en`)
- Audio format is hardcoded to 16kHz mono for Whisper compatibility
- TTS output is 24kHz (resampled from model's native rate)
- All text processing uses UTF-8 encoding

## Future Enhancements

- Direct Finnish→Portuguese translation (when better models available)
- Streaming audio processing (currently batch-only)
- Voice customization options for TTS
- Support for other language pairs
- Web interface or API server mode
