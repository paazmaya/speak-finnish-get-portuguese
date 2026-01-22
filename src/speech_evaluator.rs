use anyhow::{Context, Result};
use candle_core::{Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::whisper::{self as m, audio, Config};
use rand::distributions::{Distribution, WeightedIndex};
use std::path::Path;
use tokenizers::Tokenizer;

use crate::config;

/// Evaluator that transcribes Portuguese audio to verify TTS quality
/// This is used only in test mode to validate that the synthesized speech
/// contains the correct words. Uses a separate Portuguese Whisper model.
pub struct PortugueseEvaluator {
    device: Device,
    model: m::model::Whisper,
    tokenizer: Tokenizer,
    mel_filters: Vec<f32>,
    config: Config,
    language_token: Option<u32>,
}

impl PortugueseEvaluator {
    pub fn new(device: Device) -> Result<Self> {
        println!("🔍 Initializing Portuguese Speech Evaluator...");
        println!(
            "   Loading Portuguese Whisper model ({})...",
            config::PORTUGUESE_MODEL_ID
        );

        let model_path = config::PORTUGUESE_MODEL.path(Path::new("./models"));

        println!("   Loading from: {:?}", model_path);

        // Use local model files
        let local_config = model_path.join("config.json");
        let local_model = model_path.join("model.safetensors");
        let local_vocab = model_path.join("vocab.json");
        let local_merges = model_path.join("merges.txt");
        let local_tokenizer_config = model_path.join("tokenizer_config.json");

        // Check if model files exist
        if !local_config.exists() || !local_model.exists() {
            anyhow::bail!(
                "Portuguese evaluation model not found at {:?}\n\n\
                Please download models first using:\n  \
                cargo run --release -- --download-models",
                model_path
            );
        }

        // Load configuration from local path
        let config: Config = serde_json::from_reader(std::fs::File::open(&local_config)?)?;

        // Build tokenizer from vocab and merges files
        use tokenizers::models::bpe::BPE;
        let bpe = BPE::from_file(
            local_vocab.to_str().unwrap(),
            local_merges.to_str().unwrap(),
        )
        .build()
        .map_err(|e| anyhow::anyhow!("Failed to build BPE tokenizer: {}", e))?;
        let tokenizer = Tokenizer::new(bpe);

        // Load and apply tokenizer config
        let tokenizer_config: serde_json::Value =
            serde_json::from_reader(std::fs::File::open(&local_tokenizer_config)?)?;

        // Add special tokens if present in config
        if let Some(_special_tokens) = tokenizer_config.get("added_tokens_decoder") {
            // Whisper tokenizers typically have the special tokens already configured
            // We'll use the tokenizer as-is
        }

        // Load model from local path
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[local_model], m::DTYPE, &device)? };
        let model = m::model::Whisper::load(&vb, config.clone())?;

        // Load mel filters
        let mel_bytes = match config.num_mel_bins {
            80 => include_bytes!("melfilters.bytes").as_slice(),
            128 => include_bytes!("melfilters128.bytes").as_slice(),
            nmel => anyhow::bail!("unexpected num_mel_bins {nmel}"),
        };
        let mut mel_filters = vec![0f32; mel_bytes.len() / 4];
        byteorder::ReadBytesExt::read_f32_into::<byteorder::LittleEndian>(
            &mut &mel_bytes[..],
            &mut mel_filters,
        )?;

        // Get Portuguese language token
        let language_token = token_id(&tokenizer, "<|pt|>");

        println!("   ✓ Portuguese evaluator initialized");

        Ok(Self {
            device,
            model,
            tokenizer,
            mel_filters,
            config,
            language_token,
        })
    }

    /// Transcribe Portuguese audio to text for verification
    pub fn transcribe_portuguese(
        &mut self,
        audio_samples: &[f32],
        _models_dir: &Path,
    ) -> Result<String> {
        println!("\n🔍 EVALUATING SYNTHESIZED SPEECH");
        println!("   Transcribing Portuguese audio to verify quality...");

        // Convert audio to mel spectrogram
        let mel = audio::pcm_to_mel(&self.config, audio_samples, &self.mel_filters);
        let mel_len = mel.len();

        let mel = Tensor::from_vec(
            mel,
            (
                1,
                self.config.num_mel_bins,
                mel_len / self.config.num_mel_bins,
            ),
            &self.device,
        )?;

        // Create decoder for Portuguese
        let mut dc = Decoder::new(
            &mut self.model,
            &self.tokenizer,
            42, // seed
            &self.device,
            self.language_token,
            Task::Transcribe,
        )?;

        // Decode audio to text
        let segments = dc.run(&mel)?;

        // Combine segments into final text
        let mut result = String::new();
        for segment in segments {
            if !result.is_empty() {
                result.push(' ');
            }
            result.push_str(&segment.dr.text);
        }

        println!("   ✓ Transcribed text: \"{}\"", result);

        Ok(result.trim().to_string())
    }

    /// Compare expected text with transcribed text and provide quality metrics
    pub fn evaluate_quality(
        &mut self,
        expected: &str,
        audio_samples: &[f32],
        models_dir: &Path,
    ) -> Result<EvaluationResult> {
        let transcribed = self.transcribe_portuguese(audio_samples, models_dir)?;

        // Normalize texts for comparison (lowercase, remove punctuation)
        let normalize = |s: &str| -> String {
            s.to_lowercase()
                .chars()
                .filter(|c| c.is_alphanumeric() || c.is_whitespace())
                .collect::<String>()
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ")
        };

        let expected_norm = normalize(expected);
        let transcribed_norm = normalize(&transcribed);

        // Calculate word-level accuracy
        let expected_words: Vec<&str> = expected_norm.split_whitespace().collect();
        let transcribed_words: Vec<&str> = transcribed_norm.split_whitespace().collect();

        let mut correct_words = 0;
        let max_len = expected_words.len().max(transcribed_words.len());

        for i in 0..max_len
            .min(expected_words.len())
            .min(transcribed_words.len())
        {
            if expected_words[i] == transcribed_words[i] {
                correct_words += 1;
            }
        }

        let word_accuracy = if expected_words.is_empty() {
            0.0
        } else {
            correct_words as f32 / expected_words.len() as f32
        };

        // Check if key words are present
        let mut missing_words = Vec::new();
        for word in &expected_words {
            if !transcribed_words.contains(word) {
                missing_words.push(word.to_string());
            }
        }

        Ok(EvaluationResult {
            expected: expected.to_string(),
            transcribed,
            word_accuracy,
            missing_words,
        })
    }
}

#[derive(Debug)]
pub struct EvaluationResult {
    pub expected: String,
    pub transcribed: String,
    pub word_accuracy: f32,
    pub missing_words: Vec<String>,
}

impl EvaluationResult {
    pub fn print_report(&self) {
        println!("\n╔════════════════════════════════════════════════════════╗");
        println!("║           SPEECH QUALITY EVALUATION REPORT             ║");
        println!("╚════════════════════════════════════════════════════════╝");
        println!("\n📝 Expected Portuguese text:");
        println!("   \"{}\"", self.expected);
        println!("\n🎤 Transcribed from synthesized audio:");
        println!("   \"{}\"", self.transcribed);
        println!("\n📊 Quality Metrics:");
        println!("   Word Accuracy: {:.1}%", self.word_accuracy * 100.0);

        if self.missing_words.is_empty() {
            println!("   ✓ All expected words were correctly synthesized!");
        } else {
            println!("   ⚠️  Missing/incorrect words: {:?}", self.missing_words);
        }

        if self.word_accuracy >= 0.9 {
            println!("\n✅ EXCELLENT - Speech synthesis quality is very good!");
        } else if self.word_accuracy >= 0.7 {
            println!("\n⚠️  ACCEPTABLE - Speech synthesis is mostly correct");
        } else {
            println!("\n❌ POOR - Speech synthesis needs improvement");
        }
        println!("\n════════════════════════════════════════════════════════\n");
    }
}

fn token_id(tokenizer: &Tokenizer, token: &str) -> Option<u32> {
    tokenizer.token_to_id(token)
}

#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
enum Task {
    Transcribe,
    Translate,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct DecodingResult {
    tokens: Vec<u32>,
    text: String,
    avg_logprob: f64,
    no_speech_prob: f64,
    temperature: f64,
    compression_ratio: f64,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct Segment {
    start: f64,
    duration: f64,
    dr: DecodingResult,
}

struct Decoder<'a> {
    model: &'a mut m::model::Whisper,
    tokenizer: &'a Tokenizer,
    rng: rand::rngs::StdRng,
    device: Device,
    language_token: Option<u32>,
    task: Task,
    timestamps: bool,
    verbose: bool,
    sot_token: u32,
    transcribe_token: u32,
    translate_token: u32,
    eot_token: u32,
    no_timestamps_token: u32,
    no_speech_token: u32,
}

impl<'a> Decoder<'a> {
    fn new(
        model: &'a mut m::model::Whisper,
        tokenizer: &'a Tokenizer,
        seed: u64,
        device: &Device,
        language_token: Option<u32>,
        task: Task,
    ) -> Result<Self> {
        use rand::SeedableRng;
        let rng = rand::rngs::StdRng::seed_from_u64(seed);

        let sot_token =
            token_id(tokenizer, "<|startoftranscript|>").context("missing sot token")?;
        let transcribe_token =
            token_id(tokenizer, "<|transcribe|>").context("missing transcribe token")?;
        let translate_token =
            token_id(tokenizer, "<|translate|>").context("missing translate token")?;
        let eot_token = token_id(tokenizer, "<|endoftext|>").context("missing eot token")?;
        let no_timestamps_token =
            token_id(tokenizer, "<|notimestamps|>").context("missing no_timestamps token")?;
        let no_speech_token = m::NO_SPEECH_TOKENS
            .iter()
            .find_map(|token| token_id(tokenizer, token))
            .context("missing no_speech token")?;

        Ok(Self {
            model,
            tokenizer,
            rng,
            device: device.clone(),
            language_token,
            task,
            timestamps: false,
            verbose: false,
            sot_token,
            transcribe_token,
            translate_token,
            eot_token,
            no_timestamps_token,
            no_speech_token,
        })
    }

    fn decode(&mut self, mel: &Tensor) -> Result<DecodingResult> {
        let model = &mut self.model;
        let audio_features = model.encoder.forward(mel, true)?;

        let sample_len = model.config.max_target_positions / 2;
        let sum_logprob = 0f64;
        let mut no_speech_prob = f64::NAN;

        let mut tokens = vec![self.sot_token];
        if let Some(language_token) = self.language_token {
            tokens.push(language_token);
        }

        match self.task {
            Task::Transcribe => tokens.push(self.transcribe_token),
            Task::Translate => tokens.push(self.translate_token),
        }

        if !self.timestamps {
            tokens.push(self.no_timestamps_token);
        }

        for i in 0..sample_len {
            let tokens_t = Tensor::new(tokens.as_slice(), &self.device)?.unsqueeze(0)?;
            let ys = model.decoder.forward(&tokens_t, &audio_features, i == 0)?;

            let logits = model
                .decoder
                .final_linear(&ys.i((.., ys.dim(1)? - 1, ..))?.squeeze(1)?)?;

            if i == 0 {
                let logits = candle_nn::ops::softmax(&logits, candle_core::D::Minus1)?;
                no_speech_prob = logits
                    .i(0)?
                    .i(self.no_speech_token as usize)?
                    .to_vec0::<f32>()? as f64;
            }

            let next_token = {
                let logits = logits.to_dtype(candle_core::DType::F32)?;
                let logits_v: Vec<f32> = logits.i(0)?.to_vec1()?;

                let distr = WeightedIndex::new(&logits_v)?;
                distr.sample(&mut self.rng) as u32
            };

            tokens.push(next_token);

            if next_token == self.eot_token {
                break;
            }
        }

        let text = self
            .tokenizer
            .decode(&tokens, true)
            .map_err(anyhow::Error::msg)?;

        Ok(DecodingResult {
            tokens: tokens.clone(),
            text,
            avg_logprob: sum_logprob / tokens.len() as f64,
            no_speech_prob,
            temperature: 0.0,
            compression_ratio: 1.0,
        })
    }

    fn run(&mut self, mel: &Tensor) -> Result<Vec<Segment>> {
        let (_, _, content_frames) = mel.dims3()?;
        let mut segments = vec![];
        let mut seek = 0;

        while seek < content_frames {
            let time_offset = (seek * m::HOP_LENGTH) as f64 / m::SAMPLE_RATE as f64;
            let segment_size = usize::min(content_frames - seek, m::N_FRAMES);
            let mel_segment = mel.narrow(2, seek, segment_size)?;
            let segment_duration = (segment_size * m::HOP_LENGTH) as f64 / m::SAMPLE_RATE as f64;

            let dr = self.decode(&mel_segment)?;
            seek += segment_size;

            if dr.no_speech_prob > m::NO_SPEECH_THRESHOLD && dr.avg_logprob < m::LOGPROB_THRESHOLD {
                if self.verbose {
                    println!("No speech detected, skipping segment");
                }
                continue;
            }

            let segment = Segment {
                start: time_offset,
                duration: segment_duration,
                dr,
            };

            if self.verbose {
                println!(
                    "{:.1}s - {:.1}s: {}",
                    segment.start,
                    segment.start + segment.duration,
                    segment.dr.text
                );
            }

            segments.push(segment);
        }

        Ok(segments)
    }
}
