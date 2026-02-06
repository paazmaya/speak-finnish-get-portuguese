use anyhow::{Context, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use hound::{WavSpec, WavWriter};
use std::path::Path;
use tokenizers::Tokenizer;

use crate::qwen3_speech_tokenizer::{Qwen3SpeechTokenizerDecoder, Qwen3TTSTokenizerConfig};
use crate::qwen3_tts_model::{
    sample_token, Qwen3TTSCodePredictor, Qwen3TTSConfig, Qwen3TTSTalkerModel,
};

const DTYPE: DType = DType::F32;

pub struct PortugueseTTS {
    talker_model: Qwen3TTSTalkerModel,
    code_predictor: Qwen3TTSCodePredictor,
    speech_tokenizer: Qwen3SpeechTokenizerDecoder,
    tokenizer: Tokenizer,
    config: Qwen3TTSConfig,
    device: Device,
    sample_rate: u32,
    speaker: String,
    language: String,
}

impl PortugueseTTS {
    pub fn new(device: Device, models_dir: &Path) -> Result<Self> {
        println!("Loading Qwen3-TTS model from local directory...");

        // Model path is directly in models_dir
        let model_path = models_dir.join("Qwen--Qwen3-TTS-12Hz-0.6B-CustomVoice");

        if !model_path.exists() {
            anyhow::bail!(
                "Qwen3-TTS model not found at {:?}. Please ensure the model is downloaded.",
                model_path
            );
        }

        println!("Loading from: {:?}", model_path);

        // Load configuration
        let config_path = model_path.join("config.json");
        let config_json =
            std::fs::read_to_string(&config_path).context("Failed to read config.json")?;
        let config: Qwen3TTSConfig =
            serde_json::from_str(&config_json).context("Failed to parse config.json")?;

        println!("Model config loaded:");
        println!("  Layers: {}", config.talker_config.num_hidden_layers);
        println!("  Hidden size: {}", config.talker_config.hidden_size);
        println!("  Text hidden size: {}", config.talker_config.text_hidden_size);
        println!("  Heads: {}", config.talker_config.num_attention_heads);

        // Load tokenizer - build from vocab and merges files
        let vocab_path = model_path.join("vocab.json");
        let merges_path = model_path.join("merges.txt");
        
        // Build BPE tokenizer from vocab and merges
        let tokenizer: Result<Tokenizer> = Tokenizer::from_file(&vocab_path)
            .or_else(|_| {
                // Try to create from components if direct load fails
                use tokenizers::models::bpe::BPE;
                let bpe = BPE::from_file(&vocab_path.to_string_lossy(), &merges_path.to_string_lossy())
                    .build()
                    .map_err(|e| anyhow::anyhow!("Failed to build BPE tokenizer: {}", e))?;
                Ok(Tokenizer::new(bpe))
            })
            .map_err(|e: tokenizers::Error| anyhow::anyhow!("Failed to load tokenizer: {}", e));
        let tokenizer = tokenizer?;
        println!("Text tokenizer loaded");

        // Load main talker model weights
        let safetensors_path = model_path.join("model.safetensors");
        println!("Loading talker model weights...");
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[safetensors_path], DTYPE, &device)? };
        // Weights are stored with "talker." prefix in the SafeTensors file
        let vb = vb.pp("talker");

        let talker_model = Qwen3TTSTalkerModel::load(vb.clone(), &config.talker_config)
            .context("Failed to load talker model")?;
        println!("✓ Talker model loaded");

        println!("Loading code predictor model...");
        let code_predictor = Qwen3TTSCodePredictor::load(
            vb.pp("code_predictor"),
            &config.talker_config.code_predictor_config,
        )
        .context("Failed to load code predictor model")?;
        println!("✓ Code predictor loaded");

        // Load speech tokenizer
        let tokenizer_config_path = model_path.join("speech_tokenizer/config.json");
        let tokenizer_config_json = std::fs::read_to_string(&tokenizer_config_path)
            .context("Failed to read speech tokenizer config")?;
        let tokenizer_config: Qwen3TTSTokenizerConfig =
            serde_json::from_str(&tokenizer_config_json)
                .context("Failed to parse tokenizer config")?;

        println!("Loading speech tokenizer...");
        let tokenizer_weights_path = model_path.join("speech_tokenizer/model.safetensors");
        let tokenizer_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[tokenizer_weights_path], DTYPE, &device)?
        };

        let speech_tokenizer = Qwen3SpeechTokenizerDecoder::load(tokenizer_vb, &tokenizer_config)
            .context("Failed to load speech tokenizer")?;
        println!("✓ Speech tokenizer loaded");

        println!("Qwen3-TTS model loaded successfully");

        Ok(Self {
            talker_model,
            code_predictor,
            speech_tokenizer,
            tokenizer,
            config,
            device,
            sample_rate: 24000,          // Qwen3-TTS outputs 24kHz
            speaker: "ryan".to_string(), // Default speaker (Portuguese-capable)
            language: "portuguese".to_string(),
        })
    }

    pub fn set_speaker(&mut self, speaker: &str) {
        self.speaker = speaker.to_lowercase();
    }

    pub fn synthesize(&mut self, text: &str) -> Result<Vec<f32>> {
        println!("Generating Portuguese audio for: {}", text);
        println!("  Speaker: {}", self.speaker);
        println!("  Language: {}", self.language);

        // Tokenize text with special format
        let formatted_text = format!(
            "<|im_start|>assistant\n{}<|im_end|>\n<|im_start|>assistant\n",
            text
        );
        let encoding = self
            .tokenizer
            .encode(formatted_text, true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;

        let text_tokens = encoding.get_ids();
        println!("Text tokenized to {} tokens", text_tokens.len());

        // Get speaker and language IDs from config
        let speaker_id = self
            .config
            .talker_config
            .spk_id
            .get(&self.speaker)
            .copied()
            .ok_or_else(|| anyhow::anyhow!("Unknown speaker: {}", self.speaker))?;

        let language_id = self
            .config
            .talker_config
            .codec_language_id
            .get(&self.language)
            .copied()
            .ok_or_else(|| anyhow::anyhow!("Unknown language: {}", self.language))?;

        println!("Speaker ID: {}, Language ID: {}", speaker_id, language_id);

        let codec_prefix: Vec<u32> = vec![
            self.config.talker_config.codec_think_id,
            self.config.talker_config.codec_think_bos_id,
            language_id,
            self.config.talker_config.codec_think_eos_id,
            speaker_id,
            self.config.talker_config.codec_pad_id,
            self.config.talker_config.codec_bos_id,
        ];

        // Convert text tokens to tensor
        let text_ids = Tensor::from_vec(
            text_tokens.iter().map(|&x| x as u32).collect(),
            (1, text_tokens.len()),
            &self.device,
        )?;

        println!("Running talker model inference...");

        println!("Generating codec tokens...");

        // Increased limit for longer, more natural speech
        // At 12.5Hz frame rate: 200 tokens ≈ 16 seconds of audio
        let max_new_tokens = 200;
        let temperature = 0.7;  // Lower temperature for more stable generation
        let top_k = 50;         // Narrower sampling for better quality
        let top_p = 0.9;        // Nucleus sampling

        let mut generated_tokens = Vec::new();

        for step in 0..max_new_tokens {
            let mut codec_ids_vec = Vec::with_capacity(codec_prefix.len() + generated_tokens.len());
            codec_ids_vec.extend_from_slice(&codec_prefix);
            codec_ids_vec.extend_from_slice(&generated_tokens);

            let codec_ids = Tensor::from_vec(
                codec_ids_vec.iter().copied().collect::<Vec<u32>>(),
                (1, codec_ids_vec.len()),
                &self.device,
            )?;

            let total_len = text_tokens.len() + codec_ids_vec.len();
            let position_ids = Tensor::arange(0u32, total_len as u32, &self.device)?
                .reshape((1, total_len))?;

            let logits = self
                .talker_model
                .forward(&text_ids, &codec_ids, &position_ids)?;

            let last_logits = logits
                .i((0, total_len - 1, ..))?;

            let next_token = sample_token(&last_logits, temperature, top_k, top_p)?;

            if next_token == self.config.talker_config.codec_eos_token_id {
                println!("Reached EOS token at step {}", step);
                break;
            }

            generated_tokens.push(next_token);

            // Print every 20 tokens to show progress
            if (step + 1) % 20 == 0 {
                println!("  Generated {} tokens", step + 1);
            }
        }

        println!("Generated {} codec tokens", generated_tokens.len());

        if generated_tokens.is_empty() {
            anyhow::bail!("No codec tokens were generated by the talker model");
        }

        // Use actual number of quantizers (may be less than config for 12Hz variant)
        let num_quantizers = self.code_predictor.num_quantizers();
        let codebook_vocab = self.config.talker_config.code_predictor_config.vocab_size as u32;
        let seq_len = generated_tokens.len();

        let mut predictor_input = vec![0u32; num_quantizers * seq_len];
        for (i, &token) in generated_tokens.iter().enumerate() {
            predictor_input[i] = clamp_codebook_token(token, codebook_vocab);
        }

        let predictor_ids = Tensor::from_vec(
            predictor_input,
            (num_quantizers, seq_len),
            &self.device,
        )?;

        let predictor_position_ids = Tensor::arange(0u32, seq_len as u32, &self.device)?
            .reshape((1, seq_len))?;

        println!("Running code predictor...");
        let logits_by_group = self
            .code_predictor
            .forward(&predictor_ids, &predictor_position_ids)?;

        // Use actual number of logits returned (may be less than config for 12Hz variant)
        let actual_num_quantizers = logits_by_group.len();
        
        let mut codec_tokens_data = vec![0u32; actual_num_quantizers * seq_len];
        for (i, &token) in generated_tokens.iter().enumerate() {
            codec_tokens_data[i] = clamp_codebook_token(token, codebook_vocab);
        }

        for group_idx in 1..actual_num_quantizers {
            let logits = &logits_by_group[group_idx];
            for pos in 0..seq_len {
                let pos_logits = logits.i((0, pos, ..))?;
                // Use lower temperature for codec prediction to reduce artifacts
                let token = sample_token(&pos_logits, 0.7, 50, 0.9)?;
                codec_tokens_data[group_idx * seq_len + pos] = token;
            }
        }

        let codec_tokens = Tensor::from_vec(
            codec_tokens_data,
            (actual_num_quantizers, generated_tokens.len()),
            &self.device,
        )?;

        println!("Decoding codec tokens to audio...");

        // Decode codec tokens to waveform using speech tokenizer
        let waveform = self.speech_tokenizer.decode(&codec_tokens)?;

        println!(
            "Generated {:.2} seconds of Portuguese speech ({} samples)",
            waveform.len() as f32 / self.sample_rate as f32,
            waveform.len()
        );

        Ok(waveform)
    }

    pub fn save_wav(&self, samples: &[f32], path: impl AsRef<Path>) -> Result<()> {
        let spec = WavSpec {
            channels: 1,
            sample_rate: self.sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };

        let mut writer = WavWriter::create(path, spec)?;
        for &sample in samples {
            // Clamp to [-1.0, 1.0] range before converting
            let clamped = sample.max(-1.0).min(1.0);
            let amplitude = (clamped * i16::MAX as f32) as i16;
            writer.write_sample(amplitude)?;
        }
        writer.finalize()?;
        Ok(())
    }
}

fn clamp_codebook_token(token: u32, vocab_size: u32) -> u32 {
    if token >= vocab_size {
        0
    } else {
        token
    }
}
