use anyhow::Result;
use hf_hub::api::sync::ApiBuilder;
use std::fs;
use std::path::{Path, PathBuf};

/// Model configuration constants
pub const FINNISH_MODEL_ID: &str = "Finnish-NLP/whisper-tiny-finnish";
pub const PORTUGUESE_MODEL_ID: &str = "dominguesm/whisper-tiny-pt";
pub const KOKORO_TTS_MODEL_ID: &str = "onnx-community/Kokoro-82M-v1.0-ONNX";
pub const KOKORO_VOICES_REPO: &str = "hexgrad/Kokoro-82M";
// MADLAD-400 supports 400+ languages including Finnish to Portuguese
pub const TRANSLATION_MODEL_ID: &str = "google/madlad400-3b-mt";

/// Model configuration describing files needed for each model
pub struct ModelConfig {
    pub id: &'static str,
    pub files: &'static [&'static str],
}

impl ModelConfig {
    /// Get subdirectory name from model ID (replace / with --)
    pub fn subdir(&self) -> String {
        self.id.replace('/', "--")
    }

    /// Get full path to model directory
    pub fn path(&self, base_dir: &Path) -> PathBuf {
        base_dir.join(self.subdir())
    }

    /// Check if all required model files exist
    /// Returns Ok(()) if all files exist, Err with descriptive message otherwise
    pub fn verify_files(&self, base_dir: &Path) -> Result<()> {
        let model_path = self.path(base_dir);

        let mut missing_files = Vec::new();
        for &filename in self.files {
            let file_path = model_path.join(filename);
            if !file_path.exists() {
                missing_files.push(filename);
            }
        }

        if !missing_files.is_empty() {
            anyhow::bail!(
                "Missing model files for {}:\n  {}\n\n\
                Please download models first using:\n  \
                cargo run --release -- --download-models",
                self.id,
                missing_files.join("\n  ")
            );
        }

        Ok(())
    }
}

/// Configuration for Finnish Whisper model
pub const FINNISH_MODEL: ModelConfig = ModelConfig {
    id: FINNISH_MODEL_ID,
    files: &["config.json", "model.safetensors", "tokenizer.json"],
};

/// Configuration for Portuguese Whisper model (for evaluation)
pub const PORTUGUESE_MODEL: ModelConfig = ModelConfig {
    id: PORTUGUESE_MODEL_ID,
    files: &[
        "config.json",
        "model.safetensors",
        "vocab.json",
        "merges.txt",
        "tokenizer_config.json",
    ],
};

/// Configuration for Kokoro TTS model (Portuguese text-to-speech)
pub const KOKORO_TTS_MODEL: ModelConfig = ModelConfig {
    id: KOKORO_TTS_MODEL_ID,
    files: &[
        "onnx/model.onnx",
        "config.json",
        "tokenizer_minimal.json",
    ],
};

/// Configuration for Kokoro voices (contains voice files)
pub const KOKORO_VOICES: ModelConfig = ModelConfig {
    id: KOKORO_VOICES_REPO,
    files: &["voices/pf_dora.pt"],  // Brazilian Portuguese female voice
};

/// Configuration for MADLAD-400 Translation model (using quantized GGUF)
pub const TRANSLATION_MODEL: ModelConfig = ModelConfig {
    id: TRANSLATION_MODEL_ID,
    files: &["model-q2k.gguf", "tokenizer.json", "config.json"],
};

/// Download a model from HuggingFace to the specified directory
pub fn download_model(
    model_config: &ModelConfig,
    base_dir: &Path,
    api: &hf_hub::api::sync::Api,
) -> Result<(usize, usize)> {
    let model_path = model_config.path(base_dir);

    println!("Downloading: {}", model_config.id);
    println!("Target: {:?}\n", model_path);

    // Create model subdirectory
    fs::create_dir_all(&model_path)?;

    let repo = api.model(model_config.id.to_string());

    let mut downloaded = 0;
    let mut skipped = 0;

    for filename in model_config.files {
        let local_path = model_path.join(filename);

        if local_path.exists() {
            let file_size = fs::metadata(&local_path)?.len();
            let size_mb = file_size as f64 / (1024.0 * 1024.0);
            println!(
                "  ✓ {} already exists ({:.2} MB), skipping",
                filename, size_mb
            );
            skipped += 1;
        } else {
            // Create parent directories if the filename contains path separators
            if let Some(parent) = local_path.parent() {
                fs::create_dir_all(parent)?;
            }
            
            println!("  ⬇️  Downloading {}...", filename);
            let remote_file = repo.get(filename)?;
            fs::copy(&remote_file, &local_path)?;
            let file_size = fs::metadata(&local_path)?.len();
            let size_mb = file_size as f64 / (1024.0 * 1024.0);
            println!(
                "  ✓ {} downloaded successfully ({:.2} MB)",
                filename, size_mb
            );
            downloaded += 1;
        }
    }

    Ok((downloaded, skipped))
}

/// Download all required models
pub fn download_all_models(models_dir: &Path) -> Result<()> {
    println!("📦 Model Download Manager");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    println!("Models directory: {:?}\n", models_dir);

    // Create models directory if it doesn't exist
    fs::create_dir_all(models_dir)?;

    // Use ApiBuilder to properly configure the cache
    let api = ApiBuilder::new().with_progress(true).build()?;

    // Download Finnish model
    let (dl1, sk1) = download_model(&FINNISH_MODEL, models_dir, &api)?;

    println!();

    // Download Portuguese model
    let (dl2, sk2) = download_model(&PORTUGUESE_MODEL, models_dir, &api)?;

    println!();

    // Download Kokoro TTS model
    let (dl3, sk3) = download_model(&KOKORO_TTS_MODEL, models_dir, &api)?;

    println!();

    // Download Kokoro voices
    let (dl4, sk4) = download_model(&KOKORO_VOICES, models_dir, &api)?;

    println!();

    // Download Translation model
    let (dl5, sk5) = download_model(&TRANSLATION_MODEL, models_dir, &api)?;

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("✅ All downloads complete!");
    println!("   Downloaded: {}", dl1 + dl2 + dl3 + dl4 + dl5);
    println!("   Skipped: {}", sk1 + sk2 + sk3 + sk4 + sk5);
    println!(
        "   Total: {}\n",
        FINNISH_MODEL.files.len()
            + PORTUGUESE_MODEL.files.len()
            + KOKORO_TTS_MODEL.files.len()
            + KOKORO_VOICES.files.len()
            + TRANSLATION_MODEL.files.len()
    );

    Ok(())
}
