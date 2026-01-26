use anyhow::{Error as E, Result};
use candle_core::{DType, Device};
use hound::{WavSpec, WavWriter};
use ndarray::Array2;
use ort::session::{Session, builder::GraphOptimizationLevel};
use std::path::Path;
use tokenizers::Tokenizer;

use crate::config;

pub struct PortugueseTTS {
    session: Session,
    tokenizer: Tokenizer,
    voice_embedding: Array2<f32>,
    sample_rate: u32,
}

impl PortugueseTTS {
    pub fn new(_device: Device, models_dir: &Path) -> Result<Self> {
        println!("Loading Kokoro-82M TTS model from local directory...");

        let model_path = config::KOKORO_TTS_MODEL.path(models_dir);
        let voices_path = config::KOKORO_VOICES.path(models_dir);

        // Verify all required files exist
        config::KOKORO_TTS_MODEL.verify_files(models_dir)?;
        config::KOKORO_VOICES.verify_files(models_dir)?;

        // Use the full precision FP32 model (310MB, better quality)
        let model_file = model_path.join("onnx/model.onnx");
        let tokenizer_file = model_path.join("tokenizer_minimal.json");
        let voice_file = voices_path.join("voices/pf_dora.pt");

        println!("Loading ONNX model from: {:?}", model_file);

        // Load tokenizer
        let tokenizer = Tokenizer::from_file(tokenizer_file).map_err(E::msg)?;

        // Load ONNX model using ONNX Runtime
        let mut builder = Session::builder()?;
        builder = builder.with_optimization_level(GraphOptimizationLevel::Level3)?;
        builder = builder.with_intra_threads(4)?;
        let session = builder.commit_from_file(model_file)?;

        println!("ONNX model loaded successfully");

        // Load Brazilian Portuguese voice embedding from PyTorch file
        println!("Loading Brazilian Portuguese voice: pf_dora");
        
        let voice_embedding = match candle_core::pickle::PthTensors::new(&voice_file, None) {
            Ok(voice_tensors) => {
                let tensor_names: Vec<String> = voice_tensors.tensor_infos().keys().cloned().collect();
                println!("Found {} tensors in voice file", tensor_names.len());
                
                if tensor_names.is_empty() {
                    // Create a random voice embedding as fallback
                    println!("WARNING: No tensors in PyTorch file, using random voice embedding");
                    Array2::from_shape_fn((1, 256), |_| rand::random::<f32>())
                } else {
                    let name = &tensor_names[0];
                    println!("  Loading tensor: '{}'", name);
                    let tensor = voice_tensors.get(name)?
                        .ok_or_else(|| E::msg(format!("Failed to load tensor '{}'", name)))?
                        .to_device(&Device::Cpu)?;
                    
                    // Convert Candle tensor to ndarray
                    let shape = tensor.shape();
                    let data = tensor.to_dtype(DType::F32)?.to_vec1::<f32>()?;
                    
                    if shape.dims().len() == 1 {
                        Array2::from_shape_vec((1, shape.dims()[0]), data)?
                    } else if shape.dims().len() == 2 {
                        Array2::from_shape_vec((shape.dims()[0], shape.dims()[1]), data)?
                    } else {
                        anyhow::bail!("Unexpected voice embedding shape: {:?}", shape);
                    }
                }
            }
            Err(e) => {
                println!("WARNING: Failed to load PyTorch voice file: {}", e);
                println!("  Using random voice embedding instead");
                Array2::from_shape_fn((1, 256), |_| rand::random::<f32>())
            }
        };
        
        println!("Voice embedding shape: {:?}", voice_embedding.shape());
        println!("Kokoro-82M TTS model loaded successfully");

        Ok(Self {
            session,
            tokenizer,
            voice_embedding,
            sample_rate: 24000, // Kokoro-82M outputs 24kHz audio
        })
    }

    pub fn synthesize(&mut self, text: &str) -> Result<Vec<f32>> {
        println!("Generating Portuguese audio for: {}", text);

        // Tokenize the input text
        let tokens = self
            .tokenizer
            .encode(text, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        
        let tokens_len = tokens.len();
        println!("Text tokenized to {} tokens", tokens_len);

        // Convert tokens to i64 array for ONNX Runtime
        let input_ids: Vec<i64> = tokens.iter().map(|&x| x as i64).collect();
        let input_ids_array = Array2::from_shape_vec((1, tokens_len), input_ids)?;

        // Prepare voice embedding
        let voice_input = self.voice_embedding.clone();

        println!("Running ONNX inference...");
        println!("  Input IDs shape: {:?}", input_ids_array.shape());
        println!("  Voice embedding shape: {:?}", voice_input.shape());

        // Create ONNX Values - use tuple format (shape, vec) which is supported
        let input_shape = input_ids_array.shape();
        let voice_shape = voice_input.shape();
        
        let input_ids_value = ort::value::Value::from_array((
            [input_shape[0], input_shape[1]],
            input_ids_array.iter().copied().collect::<Vec<_>>()
        ))?;
        let voice_value = ort::value::Value::from_array((
            [voice_shape[0], voice_shape[1]],
            voice_input.iter().copied().collect::<Vec<_>>()
        ))?;
        
        // Add speed parameter (1.0 = normal speed)
        let speed_value = ort::value::Value::from_array(([1], vec![1.0f32]))?;

        // Run ONNX model inference
        let outputs = self.session.run(ort::inputs![
            "input_ids" => &input_ids_value,
            "style" => &voice_value,
            "speed" => &speed_value,
        ])?;

        // Debug: Print available output names
        println!("Available outputs:");
        for (name, _value) in outputs.iter() {
            println!("  Output: {}", name);
        }

        // Extract audio output - use the first (and likely only) output
        let (output_name, output_value) = outputs.iter().next()
            .ok_or_else(|| E::msg("No outputs from ONNX model"))?;
        
        println!("Using output: {}", output_name);
        let audio_tensor = output_value.try_extract_tensor::<f32>()?;
        let audio_data = audio_tensor.1;
        
        println!("Audio output shape: {:?}", audio_tensor.0);
        
        // Convert to Vec
        let samples: Vec<f32> = audio_data.iter().copied().collect();

        println!(
            "Generated {:.2} seconds of Portuguese speech ({} samples)",
            samples.len() as f32 / self.sample_rate as f32,
            samples.len()
        );

        Ok(samples)
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
