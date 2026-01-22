use anyhow::Result;
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::marian;
use sentencepiece::SentencePieceProcessor;
use std::path::Path;

use crate::config;

pub struct Translator {
    model_fi_en: marian::MTModel,
    model_en_pt: marian::MTModel,
    tokenizer_fi_source: SentencePieceProcessor,
    tokenizer_en_target: SentencePieceProcessor,
    tokenizer_en_source: SentencePieceProcessor,
    tokenizer_pt_target: SentencePieceProcessor,
    config_fi_en: marian::Config,
    config_en_pt: marian::Config,
    device: Device,
}

impl Translator {
    pub fn new(device: Device, models_dir: &Path) -> Result<Self> {
        // Load Stage 1: Finnish -> English model
        let model_path_fi_en = config::TRANSLATION_FI_EN_MODEL.path(models_dir);
        println!(
            "Loading Finnish->English translation model from: {:?}",
            model_path_fi_en
        );

        let config_filename_fi_en = model_path_fi_en.join("config.json");
        let weights_filename_fi_en = model_path_fi_en.join("model.safetensors");
        let source_spm_fi_en = model_path_fi_en.join("source.spm");
        let target_spm_fi_en = model_path_fi_en.join("target.spm");

        // Check FI-EN files exist
        if !config_filename_fi_en.exists()
            || !weights_filename_fi_en.exists()
            || !source_spm_fi_en.exists()
            || !target_spm_fi_en.exists()
        {
            anyhow::bail!(
                "Finnish-English translation model files not found\\n\\n\
                Please download models first using:\\n  \
                cargo run --release -- --download-models"
            );
        }

        let config_fi_en: marian::Config =
            serde_json::from_reader(std::fs::File::open(&config_filename_fi_en)?)?;

        let vb_fi_en = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[weights_filename_fi_en],
                candle_core::DType::F32,
                &device,
            )?
        };

        let model_fi_en = marian::MTModel::new(&config_fi_en, vb_fi_en)?;

        let tokenizer_fi_source =
            SentencePieceProcessor::open(&source_spm_fi_en).map_err(anyhow::Error::msg)?;
        let tokenizer_en_target =
            SentencePieceProcessor::open(&target_spm_fi_en).map_err(anyhow::Error::msg)?;

        println!("Finnish->English model loaded successfully");

        // Load Stage 2: English -> Portuguese model
        let model_path_en_pt = config::TRANSLATION_EN_PT_MODEL.path(models_dir);
        println!(
            "Loading English->Portuguese translation model from: {:?}",
            model_path_en_pt
        );

        let config_filename_en_pt = model_path_en_pt.join("config.json");
        let weights_filename_en_pt = model_path_en_pt.join("model.safetensors");
        let source_spm_en_pt = model_path_en_pt.join("source.spm");
        let target_spm_en_pt = model_path_en_pt.join("target.spm");

        // Check EN-PT files exist
        if !config_filename_en_pt.exists()
            || !weights_filename_en_pt.exists()
            || !source_spm_en_pt.exists()
            || !target_spm_en_pt.exists()
        {
            anyhow::bail!(
                "English-Portuguese translation model files not found\\n\\n\
                Please download models first using:\\n  \
                cargo run --release -- --download-models"
            );
        }

        let config_en_pt: marian::Config =
            serde_json::from_reader(std::fs::File::open(&config_filename_en_pt)?)?;

        let vb_en_pt = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[weights_filename_en_pt],
                candle_core::DType::F32,
                &device,
            )?
        };

        let model_en_pt = marian::MTModel::new(&config_en_pt, vb_en_pt)?;

        let tokenizer_en_source =
            SentencePieceProcessor::open(&source_spm_en_pt).map_err(anyhow::Error::msg)?;
        let tokenizer_pt_target =
            SentencePieceProcessor::open(&target_spm_en_pt).map_err(anyhow::Error::msg)?;

        println!("English->Portuguese model loaded successfully");
        println!("Two-stage translation ready: Finnish -> English -> Portuguese");

        Ok(Self {
            model_fi_en,
            model_en_pt,
            tokenizer_fi_source,
            tokenizer_en_target,
            tokenizer_en_source,
            tokenizer_pt_target,
            config_fi_en,
            config_en_pt,
            device,
        })
    }

    pub fn translate(&mut self, text: &str) -> Result<String> {
        println!("Translating (Stage 1 - Finnish to English): {}", text);

        // Stage 1: Finnish -> English
        // Tokenize input text using SentencePiece (source = Finnish)
        let tokens = self.tokenizer_fi_source.encode(text)?;
        let mut token_ids: Vec<u32> = tokens.iter().map(|p| p.id).collect();
        token_ids.push(self.config_fi_en.eos_token_id);

        let tokens_tensor = Tensor::new(token_ids.as_slice(), &self.device)?.unsqueeze(0)?;
        let encoder_xs = self.model_fi_en.encoder().forward(&tokens_tensor, 0)?;

        let mut decoder_token_ids = vec![self.config_fi_en.decoder_start_token_id];
        let mut english_pieces = Vec::new();

        for index in 0..512 {
            let context_size = if index >= 1 {
                1
            } else {
                decoder_token_ids.len()
            };
            let start_pos = decoder_token_ids.len().saturating_sub(context_size);
            let input_ids =
                Tensor::new(&decoder_token_ids[start_pos..], &self.device)?.unsqueeze(0)?;

            let logits = self
                .model_fi_en
                .decode(&input_ids, &encoder_xs, start_pos)?;
            let logits = logits.squeeze(0)?.get(logits.dim(0)? - 1)?;

            let token = self.sample_token(&logits)?;
            decoder_token_ids.push(token);

            if token == self.config_fi_en.eos_token_id
                || token == self.config_fi_en.forced_eos_token_id
            {
                break;
            }

            english_pieces.push(token);
        }

        let english_text = self.tokenizer_en_target.decode_piece_ids(&english_pieces)?;
        println!("  English intermediate: {}", english_text);
        println!(
            "Translating (Stage 2 - English to Portuguese): {}",
            english_text
        );

        // Stage 2: English -> Portuguese
        // Encode with English source tokenizer
        let tokens = self.tokenizer_en_source.encode(&english_text)?;
        let mut token_ids: Vec<u32> = tokens.iter().map(|p| p.id).collect();
        token_ids.push(self.config_en_pt.eos_token_id);

        let tokens_tensor = Tensor::new(token_ids.as_slice(), &self.device)?.unsqueeze(0)?;
        let encoder_xs = self.model_en_pt.encoder().forward(&tokens_tensor, 0)?;

        let mut decoder_token_ids = vec![self.config_en_pt.decoder_start_token_id];
        let mut portuguese_pieces = Vec::new();

        for index in 0..512 {
            let context_size = if index >= 1 {
                1
            } else {
                decoder_token_ids.len()
            };
            let start_pos = decoder_token_ids.len().saturating_sub(context_size);
            let input_ids =
                Tensor::new(&decoder_token_ids[start_pos..], &self.device)?.unsqueeze(0)?;

            let logits = self
                .model_en_pt
                .decode(&input_ids, &encoder_xs, start_pos)?;
            let logits = logits.squeeze(0)?.get(logits.dim(0)? - 1)?;

            let token = self.sample_token(&logits)?;
            decoder_token_ids.push(token);

            if token == self.config_en_pt.eos_token_id
                || token == self.config_en_pt.forced_eos_token_id
            {
                break;
            }

            portuguese_pieces.push(token);
        }

        let portuguese_text = self
            .tokenizer_pt_target
            .decode_piece_ids(&portuguese_pieces)?;
        Ok(portuguese_text.trim().to_string())
    }

    fn sample_token(&self, logits: &Tensor) -> Result<u32> {
        let logits = logits.to_dtype(candle_core::DType::F32)?;
        let logits_v: Vec<f32> = logits.to_vec1()?;

        let max_idx = logits_v
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();

        Ok(max_idx as u32)
    }
}
