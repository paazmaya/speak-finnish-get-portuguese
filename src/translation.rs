use anyhow::{Error as E, Result};
use candle_core::{Device, IndexOp, Tensor};
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::quantized_t5 as t5;
use std::path::Path;
use tokenizers::Tokenizer;

use crate::config;

pub struct Translator {
    model: t5::T5ForConditionalGeneration,
    tokenizer: Tokenizer,
    device: Device,
    eos_token_id: u32,
}

impl Translator {
    pub fn new(device: Device, models_dir: &Path) -> Result<Self> {
        // Load MADLAD-400 quantized GGUF model
        let model_path = config::TRANSLATION_MODEL.path(models_dir);
        println!(
            "Loading MADLAD-400 quantized translation model from: {:?}",
            model_path
        );

        // Verify all required files exist
        config::TRANSLATION_MODEL.verify_files(models_dir)?;

        let weights_filename = model_path.join("model-q2k.gguf");
        let tokenizer_filename = model_path.join("tokenizer.json");
        let config_filename = model_path.join("config.json");

        println!("Loading quantized GGUF model from: {:?}", weights_filename);
        println!("Loading config from: {:?}", config_filename);

        // Load T5 config from JSON file
        let config: t5::Config = serde_json::from_reader(std::fs::File::open(&config_filename)?)?;

        // Load GGUF using VarBuilder from quantized_var_builder
        let vb = t5::VarBuilder::from_gguf(&weights_filename, &device)?;

        // Load model from GGUF using VarBuilder
        let model = t5::T5ForConditionalGeneration::load(vb, &config)?;

        // Load tokenizer
        let tokenizer = Tokenizer::from_file(&tokenizer_filename).map_err(E::msg)?;

        // Get EOS token ID from tokenizer
        let eos_token_id = tokenizer.token_to_id("</s>").unwrap_or(1); // Default to 1 if not found

        println!("MADLAD-400 quantized model loaded successfully");
        println!("Translation ready: Finnish -> Portuguese (GGUF Q2_K quantized)");
        println!("EOS token ID: {}", eos_token_id);

        Ok(Self {
            model,
            tokenizer,
            device,
            eos_token_id,
        })
    }

    pub fn translate(&mut self, text: &str) -> Result<String> {
        println!("Translating Finnish to Portuguese: {}", text);

        // MADLAD-400 uses language codes: <2xx> for target language
        // <2pt> means translate to Portuguese
        let prompt = format!("<2pt> {}", text);

        // Tokenize input
        let tokens = self
            .tokenizer
            .encode(prompt.as_str(), true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();

        println!("  [DEBUG] Input tokens: {} tokens", tokens.len());

        let input_token_ids = Tensor::new(&tokens[..], &self.device)?.unsqueeze(0)?;

        // Encode with T5 encoder
        let encoder_output = self.model.encode(&input_token_ids)?;

        // Use pad token as decoder start (T5 standard)
        let decoder_start_token_id = 0u32; // Pad token for T5

        let mut output_token_ids = vec![decoder_start_token_id];

        // Set up logits processor for sampling
        let temperature = 0.0; // Use greedy decoding (deterministic)
        let top_p = None;
        let seed = 299792458u64; // Fixed seed for reproducibility
        let mut logits_processor = LogitsProcessor::new(seed, Some(temperature), top_p);

        let max_length = 512;

        println!("  [DEBUG] Decoder start token: {}", decoder_start_token_id);
        println!("  [DEBUG] EOS token: {}", self.eos_token_id);

        // Decode loop
        for index in 0..max_length {
            // For the first iteration, pass all tokens; for subsequent iterations, pass only the last token
            let decoder_token_ids = if index == 0 {
                Tensor::new(&output_token_ids[..], &self.device)?.unsqueeze(0)?
            } else {
                let last_token = *output_token_ids.last().unwrap();
                Tensor::new(&[last_token], &self.device)?.unsqueeze(0)?
            };

            let logits = self
                .model
                .decode(&decoder_token_ids, &encoder_output)?;

            // Debug: print logits shape
            if index == 0 {
                println!("  [DEBUG] Logits shape: {:?}", logits.dims());
            }

            // The decoder returns logits of shape [batch_size, vocab_size] for the last token
            // Just extract the batch element
            let last_logits = logits.i(0)?;

            if index == 0 {
                println!("  [DEBUG] Last logits shape: {:?}", last_logits.dims());
            }

            let next_token_id = logits_processor.sample(&last_logits)?;

            if next_token_id == self.eos_token_id {
                println!("  [DEBUG] Reached EOS at index {}", index);
                break;
            }

            output_token_ids.push(next_token_id);

            if output_token_ids.len() > max_length {
                println!("  [WARN] Reached max length");
                break;
            }
        }

        println!("  [DEBUG] Generated {} tokens", output_token_ids.len() - 1);

        // Decode tokens to text (skip the start token)
        let portuguese_text = self
            .tokenizer
            .decode(&output_token_ids[1..], true)
            .map_err(E::msg)?;

        let portuguese_text = portuguese_text.trim().to_string();

        println!("  Portuguese translation: {}", portuguese_text);

        Ok(portuguese_text)
    }
}
