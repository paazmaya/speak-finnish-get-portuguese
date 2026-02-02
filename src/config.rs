use anyhow::Result;
use hf_hub::api::sync::ApiBuilder;
use std::fs;
use std::path::{Path, PathBuf};

/// Model configuration constants
pub const QWEN3_TTS_MODEL_ID: &str = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice";
pub const TRANSLATION_MODEL_ID: &str = "google/madlad400-3b-mt";

/// Model configuration describing files needed for the model
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
                Please download the model first using:\n  \
                cargo run --release -- --download-models",
                self.id,
                missing_files.join("\n  ")
            );
        }

        Ok(())
    }
}

/// Configuration for Qwen3-TTS model (Portuguese text-to-speech)
pub const QWEN3_TTS_MODEL: ModelConfig = ModelConfig {
    id: QWEN3_TTS_MODEL_ID,
    files: &[
        "config.json",
        "model.safetensors",
        "vocab.json",
        "merges.txt",
        "tokenizer_config.json",
        "speech_tokenizer/config.json",
        "speech_tokenizer/model.safetensors",
    ],
};

/// Configuration for MADLAD-400 translation model (Finnish to Portuguese)
pub const TRANSLATION_MODEL: ModelConfig = ModelConfig {
    id: TRANSLATION_MODEL_ID,
    files: &["config.json", "model-q2k.gguf", "tokenizer.json"],
};

/// Download a single model file from HuggingFace Hub
fn download_model_file(
    api: &hf_hub::api::sync::Api,
    repo_id: &str,
    filename: &str,
    target_dir: &Path,
) -> Result<PathBuf> {
    println!("  Downloading: {}", filename);
    let repo = api.model(repo_id.to_string());
    let file = repo.get(filename)?;

    // Copy to target directory with proper structure
    let target_path = target_dir.join(filename);
    if let Some(parent) = target_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::copy(&file, &target_path)?;

    Ok(target_path)
}

/// Download Qwen3-TTS model from HuggingFace Hub
pub fn download_qwen3_model(models_dir: &Path) -> Result<()> {
    println!("\n🔽 Downloading Qwen3-TTS model...");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let target_dir = QWEN3_TTS_MODEL.path(models_dir);
    fs::create_dir_all(&target_dir)?;

    let api = ApiBuilder::new()
        .with_progress(true)
        .build()
        .map_err(|e| anyhow::anyhow!("Failed to initialize HuggingFace API: {}", e))?;

    println!("Model: {}", QWEN3_TTS_MODEL.id);
    println!("Target: {:?}\n", target_dir);

    for &filename in QWEN3_TTS_MODEL.files {
        let target_path = target_dir.join(filename);
        if target_path.exists() {
            println!("  ✓ Already exists: {}", filename);
        } else {
            download_model_file(&api, QWEN3_TTS_MODEL.id, filename, &target_dir)?;
        }
    }

    println!("\n✅ Qwen3-TTS model downloaded successfully!");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    Ok(())
}

/// Download MADLAD-400 translation model from HuggingFace Hub
pub fn download_translation_model(models_dir: &Path) -> Result<()> {
    println!("\n🔽 Downloading MADLAD-400 translation model...");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let target_dir = TRANSLATION_MODEL.path(models_dir);
    fs::create_dir_all(&target_dir)?;

    let api = ApiBuilder::new()
        .with_progress(true)
        .build()
        .map_err(|e| anyhow::anyhow!("Failed to initialize HuggingFace API: {}", e))?;

    println!("Model: {}", TRANSLATION_MODEL.id);
    println!("Target: {:?}\n", target_dir);

    for &filename in TRANSLATION_MODEL.files {
        let target_path = target_dir.join(filename);
        if target_path.exists() {
            println!("  ✓ Already exists: {}", filename);
        } else {
            download_model_file(&api, TRANSLATION_MODEL.id, filename, &target_dir)?;
        }
    }

    println!("\n✅ MADLAD-400 translation model downloaded successfully!");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    Ok(())
}
