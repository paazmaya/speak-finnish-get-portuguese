mod audio_capture;
mod config;
mod qwen3_speech_tokenizer;
mod qwen3_tts_model;
mod speech_to_text;
mod text_to_speech;
mod translation;

use anyhow::{Context, Result};
use audio_capture::AudioCapture;
use candle_core::Device;
use clap::{Parser, ValueEnum};
use cpal::traits::{DeviceTrait, HostTrait};
use crossterm::event::{self, Event, KeyCode, KeyEvent};
use speech_to_text::WhisperSpeechToText;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use text_to_speech::PortugueseTTS;
use translation::Translator;

#[derive(Parser, Debug)]
#[command(author, version, about = "Finnish to Portuguese speech translation using Qwen3-TTS", long_about = None)]
struct Args {
    /// Input text in Finnish (overrides default recording mode)
    #[arg(short, long)]
    text: Option<String>,

    /// Output audio file path
    #[arg(short, long, default_value = "output_portuguese.wav")]
    output: String,

    /// Use test mode with default Finnish text
    #[arg(long)]
    test_mode: bool,

    /// Run on CPU rather than GPU
    #[arg(long)]
    cpu: bool,

    /// Device to use for computation
    #[arg(long, value_enum, default_value = "auto")]
    device: DeviceArg,

    /// Directory containing models (for Qwen3-TTS)
    #[arg(long, default_value = "./qwen3-tts/models")]
    models_dir: PathBuf,

    /// Directory containing translation model
    #[arg(long, default_value = "./models")]
    translation_models_dir: PathBuf,

    /// Download missing models and exit
    #[arg(long)]
    download_models: bool,
}

#[derive(Debug, Clone, ValueEnum)]
enum DeviceArg {
    Auto,
    Cpu,
    Cuda,
    Metal,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Handle model download mode
    if args.download_models {
        println!("Downloading all required models...\n");
        config::download_qwen3_model(&args.models_dir)?;
        config::download_translation_model(&args.translation_models_dir)?;
        println!("\n✅ All models downloaded successfully!");
        return Ok(());
    }

    // Initialize device
    let device = match args.device {
        DeviceArg::Auto => {
            if !args.cpu {
                Device::cuda_if_available(0)?
            } else {
                Device::Cpu
            }
        }
        DeviceArg::Cpu => Device::Cpu,
        DeviceArg::Cuda => Device::new_cuda(0)?,
        DeviceArg::Metal => Device::new_metal(0)?,
    };

    println!("===== Finnish to Portuguese Speech Translation =====");
    println!("Using device: {:?}\n", device);

    // Initialize models once at startup
    println!("Loading models...");
    let mut translator = Translator::new(device.clone(), &args.translation_models_dir)?;
    let mut tts = PortugueseTTS::new(device.clone(), &args.models_dir)?;
    println!("Models loaded successfully!\n");

    // Decide mode: test mode, text mode, or default recording mode
    if args.test_mode {
        // Test mode with predefined text
        run_test_mode(&mut translator, &mut tts, &args.output)?;
    } else if let Some(text) = args.text {
        // Text mode with user-provided text
        run_text_mode(&text, &mut translator, &mut tts, &args.output)?;
    } else {
        // Default mode: space-key-triggered recording loop
        run_recording_loop(device, &args.translation_models_dir, &args.models_dir)?;
    }

    Ok(())
}

fn run_test_mode(
    translator: &mut Translator,
    tts: &mut PortugueseTTS,
    output_path: &str,
) -> Result<()> {
    let finnish_text = "Hei, miten voit tänään?";
    println!("Running in test mode with Finnish text: {}\n", finnish_text);

    let portuguese_text = translator.translate(finnish_text)?;
    println!("Portuguese translation: {}\n", portuguese_text);

    println!("Synthesizing Portuguese speech...");
    let audio_output = tts.synthesize(&portuguese_text)?;
    tts.save_wav(&audio_output, output_path)?;

    println!("\n===== Translation Complete =====");
    println!("Finnish: {}", finnish_text);
    println!("Portuguese: {}", portuguese_text);
    println!("Audio saved to: {}", output_path);

    Ok(())
}

fn run_text_mode(
    finnish_text: &str,
    translator: &mut Translator,
    tts: &mut PortugueseTTS,
    output_path: &str,
) -> Result<()> {
    println!("Finnish input text: {}\n", finnish_text);

    let portuguese_text = translator.translate(finnish_text)?;
    println!("Portuguese translation: {}\n", portuguese_text);

    println!("Synthesizing Portuguese speech...");
    let audio_output = tts.synthesize(&portuguese_text)?;
    tts.save_wav(&audio_output, output_path)?;

    println!("\n===== Translation Complete =====");
    println!("Finnish: {}", finnish_text);
    println!("Portuguese: {}", portuguese_text);
    println!("Audio saved to: {}", output_path);

    Ok(())
}

fn run_recording_loop(
    device: Device,
    translation_models_dir: &PathBuf,
    tts_models_dir: &PathBuf,
) -> Result<()> {
    println!("===== Space-Key Recording Mode =====");
    println!("Press SPACE to start/stop recording");
    println!("Press 'q' or ESC to quit\n");

    // Initialize models
    let mut stt = WhisperSpeechToText::new(device.clone(), translation_models_dir)?;
    let mut translator = Translator::new(device.clone(), translation_models_dir)?;
    let mut tts = PortugueseTTS::new(device.clone(), tts_models_dir)?;

    // Setup audio stream variables outside loop
    let host = cpal::default_host();
    let audio_device = host
        .default_input_device()
        .context("No input device available")?;

    let config = audio_device
        .default_input_config()
        .context("Failed to get default input config")?;

    let sample_rate = config.sample_rate().0;
    let channels = config.channels() as usize;

    // Enable raw mode for keyboard input
    crossterm::terminal::enable_raw_mode()?;

    let mut recording_count = 0;
    let mut is_recording = false;
    let recorded_samples = Arc::new(Mutex::new(Vec::new()));

    println!("🎤 Ready! Press SPACE to start recording...\n");

    let mut stream_option: Option<cpal::platform::Stream> = None;

    loop {
        // Poll for keyboard events
        if event::poll(Duration::from_millis(100))? {
            if let Event::Key(KeyEvent { code, .. }) = event::read()? {
                match code {
                    KeyCode::Char(' ') => {
                        if !is_recording {
                            // Start recording
                            is_recording = true;
                            recorded_samples.lock().unwrap().clear();

                            let samples_clone = recorded_samples.clone();

                            // Build and start stream
                            let stream = match config.sample_format() {
                                cpal::SampleFormat::F32 => audio_device.build_input_stream(
                                    &config.clone().into(),
                                    move |data: &[f32], _: &cpal::InputCallbackInfo| {
                                        let mut samples = samples_clone.lock().unwrap();
                                        samples.extend_from_slice(data);
                                    },
                                    |err| eprintln!("Audio stream error: {}", err),
                                    None,
                                )?,
                                cpal::SampleFormat::I16 => audio_device.build_input_stream(
                                    &config.clone().into(),
                                    move |data: &[i16], _: &cpal::InputCallbackInfo| {
                                        let mut samples = samples_clone.lock().unwrap();
                                        for &sample in data {
                                            samples.push(sample as f32 / i16::MAX as f32);
                                        }
                                    },
                                    |err| eprintln!("Audio stream error: {}", err),
                                    None,
                                )?,
                                cpal::SampleFormat::U16 => audio_device.build_input_stream(
                                    &config.clone().into(),
                                    move |data: &[u16], _: &cpal::InputCallbackInfo| {
                                        let mut samples = samples_clone.lock().unwrap();
                                        for &sample in data {
                                            samples.push((sample as f32 / u16::MAX as f32) * 2.0 - 1.0);
                                        }
                                    },
                                    |err| eprintln!("Audio stream error: {}", err),
                                    None,
                                )?,
                                _ => {
                                    crossterm::terminal::disable_raw_mode()?;
                                    anyhow::bail!("Unsupported sample format: {:?}", config.sample_format())
                                }
                            };

                            use cpal::traits::StreamTrait;
                            stream.play()?;
                            stream_option = Some(stream);

                            println!("🔴 Recording... Press SPACE to stop");
                        } else {
                            // Stop recording
                            is_recording = false;
                            drop(stream_option.take());

                            let mut samples = recorded_samples.lock().unwrap().clone();

                            if samples.is_empty() {
                                println!("⚠️  No audio recorded, please try again\n");
                                println!("🎤 Press SPACE to start recording...");
                                continue;
                            }

                            recording_count += 1;
                            println!("\n⏹️  Recording stopped. Processing audio...\n");

                            // Convert to mono if stereo
                            if channels == 2 {
                                samples = samples
                                    .chunks_exact(2)
                                    .map(|chunk| (chunk[0] + chunk[1]) / 2.0)
                                    .collect();
                            }

                            // Resample to 16kHz if needed
                            if sample_rate != 16000 {
                                samples = resample(&samples, sample_rate, 16000)?;
                            }

                            // Process the audio through the pipeline
                            match process_audio_pipeline(&samples, &mut stt, &mut translator, &mut tts) {
                                Ok((finnish, portuguese)) => {
                                    println!("\n✅ Translation #{} complete!", recording_count);
                                    println!("Finnish: {}", finnish);
                                    println!("Portuguese: {}", portuguese);
                                    println!("\n🎤 Press SPACE to record again, or 'q' to quit...\n");
                                }
                                Err(e) => {
                                    println!("\n❌ Error processing audio: {}", e);
                                    println!("🎤 Press SPACE to try again, or 'q' to quit...\n");
                                }
                            }
                        }
                    }
                    KeyCode::Char('q') | KeyCode::Esc => {
                        println!("\n\n🛑 Quitting...");
                        break;
                    }
                    _ => {}
                }
            }
        }
    }

    // Cleanup
    drop(stream_option);
    crossterm::terminal::disable_raw_mode()?;

    println!("Total recordings processed: {}", recording_count);
    println!("Goodbye!");

    Ok(())
}

fn process_audio_pipeline(
    audio_samples: &[f32],
    stt: &mut WhisperSpeechToText,
    translator: &mut Translator,
    tts: &mut PortugueseTTS,
) -> Result<(String, String)> {
    // Step 1: Transcribe Finnish speech to text
    println!("Step 1: Transcribing Finnish speech...");
    let finnish_text = stt.transcribe(audio_samples)?;
    println!("Transcribed: {}", finnish_text);

    // Step 2: Translate to Portuguese
    println!("\nStep 2: Translating to Portuguese...");
    let portuguese_text = translator.translate(&finnish_text)?;
    println!("Translated: {}", portuguese_text);

    // Step 3: Synthesize Portuguese speech
    println!("\nStep 3: Synthesizing Portuguese speech...");
    let audio_output = tts.synthesize(&portuguese_text)?;

    // Step 4: Play the audio (save to temp file and play)
    let temp_output = format!("output_{}.wav", std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_secs());
    tts.save_wav(&audio_output, &temp_output)?;

    // Play the audio file
    println!("🔊 Playing Portuguese audio...");
    play_audio(&temp_output)?;

    Ok((finnish_text, portuguese_text))
}

fn resample(samples: &[f32], from_rate: u32, to_rate: u32) -> Result<Vec<f32>> {
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

fn play_audio(file_path: &str) -> Result<()> {
    use std::process::Command;

    #[cfg(target_os = "macos")]
    {
        Command::new("afplay")
            .arg(file_path)
            .status()
            .context("Failed to play audio with afplay")?;
    }

    #[cfg(target_os = "linux")]
    {
        Command::new("aplay")
            .arg(file_path)
            .status()
            .context("Failed to play audio with aplay")?;
    }

    #[cfg(target_os = "windows")]
    {
        // Windows doesn't have a simple command-line player by default
        // Could use powershell or add a dependency for audio playback
        println!("Audio saved to: {}", file_path);
        println!("Please play manually or use a media player");
    }

    Ok(())
}
