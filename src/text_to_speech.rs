use anyhow::{Error as E, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::parler_tts::{Config, Model};
use hf_hub::api::sync::Api;
use hound::{WavSpec, WavWriter};
use std::path::Path;
use tokenizers::Tokenizer;

pub struct PortugueseTTS {
    model: Model,
    tokenizer: Tokenizer,
    config: Config,
    device: Device,
}

impl PortugueseTTS {
    pub fn new(device: Device) -> Result<Self> {
        println!("Loading Portuguese Parler-TTS model from HuggingFace Hub...");

        let api = Api::new()?;
        let model_id = "freds0/parler-tts-mini-v1.1-ptbr";
        let revision = "main";

        let repo = api.repo(hf_hub::Repo::with_revision(
            model_id.to_string(),
            hf_hub::RepoType::Model,
            revision.to_string(),
        ));

        // Load model files
        let model_file = repo.get("model.safetensors")?;
        let config_file = repo.get("config.json")?;
        let tokenizer_file = repo.get("tokenizer.json")?;

        println!("Retrieved model files");

        // Load tokenizer
        let tokenizer = Tokenizer::from_file(tokenizer_file).map_err(E::msg)?;

        // Load config
        let config: Config = serde_json::from_reader(std::fs::File::open(config_file)?)?;

        // Load model
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[model_file], DType::F32, &device)? };
        let model = Model::new(&config, vb)?;

        println!("Portuguese TTS model loaded successfully");

        Ok(Self {
            model,
            tokenizer,
            config,
            device,
        })
    }

    pub fn synthesize(&mut self, text: &str) -> Result<Vec<f32>> {
        println!("Generating Portuguese audio for: {}", text);

        // Default Portuguese description for natural female voice
        let description = "Uma voz feminina clara e expressiva, falando em português brasileiro com velocidade e tom moderados. A gravação é de alta qualidade, com a voz soando próxima e natural.";

        // Tokenize description
        let description_tokens = self
            .tokenizer
            .encode(description, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        let description_tokens = Tensor::new(description_tokens, &self.device)?.unsqueeze(0)?;

        // Tokenize prompt (text to synthesize)
        let prompt_tokens = self
            .tokenizer
            .encode(text, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        let prompt_tokens = Tensor::new(prompt_tokens, &self.device)?.unsqueeze(0)?;

        // Create logits processor
        let lp = candle_transformers::generation::LogitsProcessor::new(
            42,        // seed
            Some(0.0), // temperature (0.0 for deterministic)
            None,      // top_p
        );

        println!("Generating audio codes...");
        let max_steps = 512;
        let codes = self
            .model
            .generate(&prompt_tokens, &description_tokens, lp, max_steps)?;

        // Decode codes to audio
        println!("Decoding audio...");
        let codes = codes.to_dtype(DType::I64)?.unsqueeze(0)?;
        let pcm = self
            .model
            .audio_encoder
            .decode_codes(&codes.to_device(&self.device)?)?;

        let pcm = pcm.i((0, 0))?;

        // Normalize loudness
        let sample_rate = self.config.audio_encoder.sampling_rate as usize;
        let pcm = normalize_loudness(&pcm, sample_rate, true)?;

        let samples = pcm.to_vec1::<f32>()?;

        println!(
            "Generated {:.2} seconds of Portuguese speech",
            samples.len() as f32 / sample_rate as f32
        );

        Ok(samples)
    }

    pub fn save_wav(&self, samples: &[f32], path: impl AsRef<Path>) -> Result<()> {
        let sample_rate = self.config.audio_encoder.sampling_rate;
        let spec = WavSpec {
            channels: 1,
            sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };

        let mut writer = WavWriter::create(path, spec)?;
        for &sample in samples {
            let amplitude = (sample * i16::MAX as f32) as i16;
            writer.write_sample(amplitude)?;
        }
        writer.finalize()?;
        Ok(())
    }
}

// Audio normalization helper function
fn normalize_loudness(
    audio: &Tensor,
    _sample_rate: usize,
    apply_compressor: bool,
) -> Result<Tensor> {
    let audio_data = audio.to_vec1::<f32>()?;

    // Calculate RMS (Root Mean Square) for loudness
    let rms: f32 = audio_data.iter().map(|&x| x * x).sum::<f32>() / audio_data.len() as f32;
    let rms = rms.sqrt();

    // Target RMS for normalization (-20 dBFS)
    let target_rms = 0.1;

    let normalized: Vec<f32> = if rms > 0.0 {
        let gain = target_rms / rms;
        audio_data.iter().map(|&x| x * gain).collect()
    } else {
        audio_data.clone()
    };

    // Apply simple compression if requested
    let final_audio = if apply_compressor {
        normalized
            .iter()
            .map(|&x| {
                // Soft clipping to prevent distortion
                let threshold = 0.8;
                if x.abs() > threshold {
                    threshold * x.signum() + (x - threshold * x.signum()) * 0.5
                } else {
                    x
                }
            })
            .collect()
    } else {
        normalized
    };

    Ok(Tensor::new(final_audio, audio.device())?)
}
