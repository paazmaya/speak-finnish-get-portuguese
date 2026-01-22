use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use crossterm::event::{self, Event, KeyCode, KeyEvent};
use hound::WavReader;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Audio capture and processing for microphone input and WAV files
pub struct AudioCapture;

impl AudioCapture {
    pub fn new() -> Self {
        Self
    }

    /// Load and decode a WAV file to 16kHz mono f32 samples
    pub fn load_wav(&self, path: impl AsRef<Path>) -> Result<Vec<f32>> {
        let mut reader = WavReader::open(path.as_ref())
            .with_context(|| format!("Failed to open WAV file: {:?}", path.as_ref()))?;

        let spec = reader.spec();
        println!("Loaded WAV: {:?}", spec);

        if spec.sample_rate != 16000 {
            anyhow::bail!(
                "Expected 16kHz sample rate, got {}Hz. Please resample your audio file.",
                spec.sample_rate
            );
        }

        let samples: Result<Vec<f32>> = match spec.sample_format {
            hound::SampleFormat::Float => reader
                .samples::<f32>()
                .map(|s| s.map_err(|e| anyhow::anyhow!("Sample read error: {}", e)))
                .collect(),
            hound::SampleFormat::Int => {
                let max_val = (1i32 << (spec.bits_per_sample - 1)) as f32;
                reader
                    .samples::<i32>()
                    .map(|s| {
                        s.map(|sample| sample as f32 / max_val)
                            .map_err(|e| anyhow::anyhow!("Sample read error: {}", e))
                    })
                    .collect()
            }
        };

        let mut samples = samples?;

        // Convert to mono if stereo
        if spec.channels == 2 {
            samples = samples
                .chunks_exact(2)
                .map(|chunk| (chunk[0] + chunk[1]) / 2.0)
                .collect();
        }

        println!("Loaded {} samples", samples.len());
        Ok(samples)
    }

    /// Save f32 audio samples to a 16kHz mono WAV file
    pub fn save_wav(&self, samples: &[f32], path: impl AsRef<Path>) -> Result<()> {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 16000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };

        let mut writer = hound::WavWriter::create(path.as_ref(), spec)?;

        for &sample in samples {
            let amplitude = (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
            writer.write_sample(amplitude)?;
        }

        writer.finalize()?;
        println!("Audio saved to: {:?}", path.as_ref());

        Ok(())
    }

    /// Record audio from microphone for a fixed duration
    /// Returns f32 samples at 16kHz mono
    pub fn record_from_microphone(&self, duration_secs: u64) -> Result<Vec<f32>> {
        println!("Initializing audio capture...");

        // Use default host (JACK should be default when running)
        let host = cpal::default_host();

        // Get the default input device
        let device = host
            .default_input_device()
            .context("No input device available")?;

        println!("Using input device: {}", device.name()?);

        // Get the default input config
        let config = device
            .default_input_config()
            .context("Failed to get default input config")?;

        println!("Input config: {:?}", config);

        let sample_rate = config.sample_rate().0;
        let channels = config.channels() as usize;

        // Shared buffer to collect samples
        let samples = Arc::new(Mutex::new(Vec::new()));
        let samples_clone = samples.clone();

        // Build the input stream
        let stream = match config.sample_format() {
            cpal::SampleFormat::F32 => device.build_input_stream(
                &config.into(),
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    let mut samples = samples_clone.lock().unwrap();
                    samples.extend_from_slice(data);
                },
                |err| eprintln!("Audio stream error: {}", err),
                None,
            )?,
            cpal::SampleFormat::I16 => device.build_input_stream(
                &config.into(),
                move |data: &[i16], _: &cpal::InputCallbackInfo| {
                    let mut samples = samples_clone.lock().unwrap();
                    for &sample in data {
                        samples.push(sample as f32 / i16::MAX as f32);
                    }
                },
                |err| eprintln!("Audio stream error: {}", err),
                None,
            )?,
            cpal::SampleFormat::U16 => {
                device.build_input_stream(
                    &config.into(),
                    move |data: &[u16], _: &cpal::InputCallbackInfo| {
                        let mut samples = samples_clone.lock().unwrap();
                        for &sample in data {
                            // Convert u16 to f32 in range [-1.0, 1.0]
                            samples.push((sample as f32 / u16::MAX as f32) * 2.0 - 1.0);
                        }
                    },
                    |err| eprintln!("Audio stream error: {}", err),
                    None,
                )?
            }
            _ => anyhow::bail!("Unsupported sample format: {:?}", config.sample_format()),
        };

        // Start recording
        stream.play()?;
        println!("Recording for {} seconds... Speak now!", duration_secs);

        // Record for the specified duration
        std::thread::sleep(Duration::from_secs(duration_secs));

        // Stop the stream
        drop(stream);
        println!("Recording finished.");

        // Get the recorded samples
        let mut recorded_samples = samples.lock().unwrap().clone();

        println!(
            "Recorded {} samples at {}Hz with {} channels",
            recorded_samples.len(),
            sample_rate,
            channels
        );

        // Convert to mono if stereo
        if channels == 2 {
            recorded_samples = recorded_samples
                .chunks_exact(2)
                .map(|chunk| (chunk[0] + chunk[1]) / 2.0)
                .collect();
            println!("Converted to mono: {} samples", recorded_samples.len());
        }

        // Resample to 16kHz if needed
        if sample_rate != 16000 {
            println!("Resampling from {}Hz to 16000Hz...", sample_rate);
            recorded_samples = self.resample(&recorded_samples, sample_rate, 16000)?;
            println!("Resampled to {} samples", recorded_samples.len());
        }

        Ok(recorded_samples)
    }

    /// Resample audio using linear interpolation
    fn resample(&self, samples: &[f32], from_rate: u32, to_rate: u32) -> Result<Vec<f32>> {
        if from_rate == to_rate {
            return Ok(samples.to_vec());
        }

        let ratio = from_rate as f64 / to_rate as f64;
        let output_len = (samples.len() as f64 / ratio) as usize;
        let mut output = Vec::with_capacity(output_len);

        for i in 0..output_len {
            let src_idx = i as f64 * ratio;
            let idx0 = src_idx.floor() as usize;
            let idx1 = (idx0 + 1).min(samples.len() - 1);
            let frac = src_idx - idx0 as f64;

            let sample = samples[idx0] * (1.0 - frac) as f32 + samples[idx1] * frac as f32;
            output.push(sample);
        }

        Ok(output)
    }

    /// Record audio in segments controlled by space key
    /// Press SPACE to cut segment, 'q' or ESC to quit
    /// Returns iterator of audio segments as 16kHz mono f32 samples
    pub fn record_segments_with_space_key(&self) -> Result<impl Iterator<Item = Vec<f32>>> {
        println!("Initializing segmented audio capture...");
        println!("Press SPACE to save current segment and start a new one");
        println!("Press 'q' or ESC to quit recording\n");

        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .context("No input device available")?;

        println!("Using input device: {}", device.name()?);

        let config = device
            .default_input_config()
            .context("Failed to get default input config")?;

        println!("Input config: {:?}", config);

        let sample_rate = config.sample_rate().0;
        let channels = config.channels() as usize;

        // Shared buffer to collect samples
        let samples = Arc::new(Mutex::new(Vec::new()));
        let samples_clone = samples.clone();

        // Build the input stream
        let stream = match config.sample_format() {
            cpal::SampleFormat::F32 => device.build_input_stream(
                &config.into(),
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    let mut samples = samples_clone.lock().unwrap();
                    samples.extend_from_slice(data);
                },
                |err| eprintln!("Audio stream error: {}", err),
                None,
            )?,
            cpal::SampleFormat::I16 => device.build_input_stream(
                &config.into(),
                move |data: &[i16], _: &cpal::InputCallbackInfo| {
                    let mut samples = samples_clone.lock().unwrap();
                    for &sample in data {
                        samples.push(sample as f32 / i16::MAX as f32);
                    }
                },
                |err| eprintln!("Audio stream error: {}", err),
                None,
            )?,
            cpal::SampleFormat::U16 => device.build_input_stream(
                &config.into(),
                move |data: &[u16], _: &cpal::InputCallbackInfo| {
                    let mut samples = samples_clone.lock().unwrap();
                    for &sample in data {
                        samples.push((sample as f32 / u16::MAX as f32) * 2.0 - 1.0);
                    }
                },
                |err| eprintln!("Audio stream error: {}", err),
                None,
            )?,
            _ => anyhow::bail!("Unsupported sample format: {:?}", config.sample_format()),
        };

        // Start recording
        stream.play()?;
        println!("🎤 Recording started... Speak and press SPACE when done with segment.");

        let mut segments = Vec::new();
        let mut segment_count = 0;

        // Enable raw mode for keyboard input
        crossterm::terminal::enable_raw_mode()?;

        loop {
            // Poll for keyboard events with timeout
            if event::poll(Duration::from_millis(100))? {
                if let Event::Key(KeyEvent { code, .. }) = event::read()? {
                    match code {
                        KeyCode::Char(' ') => {
                            // Extract current segment
                            let mut current_samples = samples.lock().unwrap().clone();
                            samples.lock().unwrap().clear();

                            if !current_samples.is_empty() {
                                segment_count += 1;
                                println!(
                                    "\n✂️  Segment {} saved ({} samples)",
                                    segment_count,
                                    current_samples.len()
                                );
                                println!("🎤 Ready for next segment... Press SPACE when done or 'q' to quit.");

                                // Convert to mono if stereo
                                if channels == 2 {
                                    current_samples = current_samples
                                        .chunks_exact(2)
                                        .map(|chunk| (chunk[0] + chunk[1]) / 2.0)
                                        .collect();
                                }

                                // Resample to 16kHz if needed
                                if sample_rate != 16000 {
                                    current_samples =
                                        self.resample(&current_samples, sample_rate, 16000)?;
                                }

                                segments.push(current_samples);
                            } else {
                                println!("\n⚠️  No audio recorded yet, keep speaking...");
                            }
                        }
                        KeyCode::Char('q') | KeyCode::Esc => {
                            println!("\n\n🛑 Stopping recording...");
                            break;
                        }
                        _ => {}
                    }
                }
            }
        }

        // Cleanup
        crossterm::terminal::disable_raw_mode()?;
        drop(stream);

        println!("Recording finished. Total segments: {}", segments.len());

        Ok(segments.into_iter())
    }
}
