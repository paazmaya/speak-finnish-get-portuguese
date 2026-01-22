use anyhow::{Context, Result};
use candle_core::{Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::whisper::{self as m, audio, Config};
use rand::distributions::{Distribution, WeightedIndex};
use std::path::Path;
use tokenizers::Tokenizer;

use crate::config;

pub struct WhisperSpeechToText {
    model: m::model::Whisper,
    tokenizer: Tokenizer,
    mel_filters: Vec<f32>,
    device: Device,
    config: Config,
    language_token: Option<u32>,
}

impl WhisperSpeechToText {
    pub fn new(device: Device, models_dir: &Path) -> Result<Self> {
        // Use subdirectory for Finnish model
        let model_path = config::FINNISH_MODEL.path(models_dir);
        println!("Loading Whisper model from: {:?}", model_path);

        // Verify all required files exist
        config::FINNISH_MODEL.verify_files(models_dir)?;

        let config_filename = model_path.join("config.json");
        let tokenizer_filename = model_path.join("tokenizer.json");
        let weights_filename = model_path.join("model.safetensors");

        let config: Config = serde_json::from_reader(std::fs::File::open(&config_filename)?)?;
        let tokenizer = Tokenizer::from_file(&tokenizer_filename).map_err(anyhow::Error::msg)?;

        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], m::DTYPE, &device)? };

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

        // Get Finnish language token
        let language_token = token_id(&tokenizer, "<|fi|>");

        println!("Loaded Finnish-specific Whisper model (RASMUS/whisper-small-fi)");

        Ok(Self {
            model,
            tokenizer,
            mel_filters,
            device,
            config,
            language_token,
        })
    }

    pub fn transcribe(&mut self, audio_data: &[f32]) -> Result<String> {
        let mel = audio::pcm_to_mel(&self.config, audio_data, &self.mel_filters);
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

        println!("Mel spectrogram shape: {:?}", mel.dims());

        let mut dc = Decoder::new(
            &mut self.model,
            &self.tokenizer,
            299792458, // seed
            &self.device,
            self.language_token,
            Task::Transcribe,
        )?;

        let segments = dc.run(&mel)?;

        let mut text = String::new();
        for segment in segments {
            text.push_str(&segment.dr.text);
        }

        Ok(text.trim().to_string())
    }
}

pub fn token_id(tokenizer: &Tokenizer, token: &str) -> Option<u32> {
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
