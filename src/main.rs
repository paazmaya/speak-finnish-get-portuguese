mod audio_capture;
mod config;
mod speech_evaluator;
mod speech_to_text;
mod text_to_speech;

mod translation;

use anyhow::Result;
use candle_core::Device;
use clap::{Parser, ValueEnum};
use std::path::Path;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about = "Finnish to Portuguese speech translation using Candle", long_about = None)]
struct Args {
    /// Input audio file (WAV format, 16kHz, mono)
    #[arg(short, long)]
    input: Option<PathBuf>,

    /// Output audio file path
    #[arg(short, long, default_value = "output_portuguese.wav")]
    output: String,

    /// Save input audio to file (when recording from microphone)
    #[arg(long)]
    save_input: Option<String>,

    /// Record from microphone (default mode if no input file specified)
    #[arg(short, long)]
    microphone: bool,

    /// Recording duration in seconds (only used with --microphone)
    #[arg(short, long, default_value = "5")]
    duration: u64,

    /// Enable segmented recording mode with space key to cut segments
    #[arg(long)]
    segmented: bool,

    /// Show the original Finnish transcription text
    #[arg(short = 's', long)]
    show_original: bool,

    /// Run on CPU rather than GPU
    #[arg(long)]
    cpu: bool,

    /// Device to use for computation
    #[arg(long, value_enum, default_value = "auto")]
    device: DeviceArg,

    /// Use test text instead of audio file (set via --test-text)
    #[arg(long)]
    test_mode: bool,

    /// Test text to translate (only used with --test-mode)
    #[arg(long, default_value = "Hei, miten voit tänään?")]
    test_text: String,

    /// Directory containing model files
    #[arg(long, default_value = "./models")]
    models_dir: PathBuf,

    /// Download missing models and exit
    #[arg(long)]
    download_models: bool,

    /// Evaluate synthesized speech quality (test mode only)
    #[arg(long)]
    evaluate: bool,
}

#[derive(Debug, Clone, ValueEnum)]
enum DeviceArg {
    Auto,
    Cpu,
    Cuda,
    Metal,
}

/// Load audio from file or record from microphone
fn get_audio_samples(args: &Args) -> Result<Vec<f32>> {
    if let Some(input_path) = &args.input {
        println!("Step 1: Loading audio from {:?}...", input_path);
        let audio_capture = audio_capture::AudioCapture::new();
        let audio_samples = audio_capture.load_wav(input_path)?;
        println!("Loaded {} samples\n", audio_samples.len());

        if let Some(input_save_path) = &args.save_input {
            println!("Saving input audio to {:?}...", input_save_path);
            audio_capture.save_wav(&audio_samples, input_save_path)?;
        }

        Ok(audio_samples)
    } else if args.microphone {
        println!("Step 1: Recording from microphone...");
        let audio_capture = audio_capture::AudioCapture::new();
        let audio_samples = audio_capture.record_from_microphone(args.duration)?;
        println!("Recorded {} samples\n", audio_samples.len());

        if let Some(input_save_path) = &args.save_input {
            println!("Saving input audio to {:?}...", input_save_path);
            audio_capture.save_wav(&audio_samples, input_save_path)?;
        }

        Ok(audio_samples)
    } else {
        anyhow::bail!("Please provide either --input <file.wav>, --microphone, or --test-mode");
    }
}

/// Transcribe audio samples to Finnish text using Whisper
fn transcribe_audio(
    audio_samples: &[f32],
    device: &Device,
    show_original: bool,
    models_dir: &Path,
) -> Result<String> {
    println!("Step 2: Transcribing Finnish speech...");
    let mut stt = speech_to_text::WhisperSpeechToText::new(device.clone(), models_dir)?;
    let finnish_text = stt.transcribe(audio_samples)?;

    if show_original {
        println!("\n========================================");
        println!("ORIGINAL FINNISH TEXT:");
        println!("========================================");
        println!("{}", finnish_text);
        println!("========================================\n");
    } else {
        println!("Transcribed Finnish text: {}\n", finnish_text);
    }

    Ok(finnish_text)
}

/// Translate Finnish text to Portuguese using Marian MT
fn translate_text(finnish_text: &str, device: &Device, models_dir: &Path) -> Result<String> {
    println!("Step 3: Translating from Finnish to Portuguese...");
    let mut translator = translation::Translator::new(device.clone(), models_dir)?;
    let portuguese_text = translator.translate(finnish_text)?;
    println!("Translated Portuguese text: {}\n", portuguese_text);
    Ok(portuguese_text)
}

/// Synthesize Portuguese speech from text using TTS
fn synthesize_speech(
    portuguese_text: &str,
    device: &Device,
    models_dir: &Path,
) -> Result<Vec<f32>> {
    println!("Step 4: Synthesizing Portuguese speech...");
    let mut tts = text_to_speech::PortugueseTTS::new(device.clone(), models_dir)?;
    let audio_output = tts.synthesize(portuguese_text)?;
    Ok(audio_output)
}

/// Save audio samples to WAV file
fn save_output_audio(
    audio_output: &[f32],
    output_path: &str,
    device: &Device,
    models_dir: &Path,
) -> Result<()> {
    println!("Step 5: Saving output audio...");
    let tts = text_to_speech::PortugueseTTS::new(device.clone(), models_dir)?;
    tts.save_wav(audio_output, output_path)?;
    Ok(())
}

fn print_results(
    finnish_text: &str,
    portuguese_text: &str,
    output_path: &str,
    show_original: bool,
) {
    println!("\n===== Translation Pipeline Complete =====");
    if show_original {
        println!("\n📝 ORIGINAL (Finnish):\n   {}", finnish_text);
        println!("\n🇵🇹 TRANSLATED (Portuguese):\n   {}", portuguese_text);
    } else {
        println!("Finnish: {}", finnish_text);
        println!("Portuguese: {}", portuguese_text);
    }
    println!("Audio saved to: {}", output_path);
}

/// Download missing models to the specified directory
fn download_models(models_dir: &Path) -> Result<()> {
    config::download_all_models(models_dir)
}

/// Process segmented recording with space key to cut segments
/// Each segment is processed and saved with an incremental filename
fn process_segmented_recording(args: &Args, device: &Device) -> Result<()> {
    println!("🎯 SEGMENTED RECORDING MODE");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let audio_capture = audio_capture::AudioCapture::new();
    let segments = audio_capture.record_segments_with_space_key()?;

    let mut segment_num = 0;
    for audio_samples in segments {
        segment_num += 1;
        println!("\n╔════════════════════════════════════════╗");
        println!("║      Processing Segment #{}           ║", segment_num);
        println!("╚════════════════════════════════════════╝\n");

        // Transcribe
        let finnish_text =
            transcribe_audio(&audio_samples, device, args.show_original, &args.models_dir)?;

        // Translate
        let portuguese_text = translate_text(&finnish_text, device, &args.models_dir)?;

        // Synthesize
        let audio_output = synthesize_speech(&portuguese_text, device, &args.models_dir)?;

        // Save with incremental filename
        let output_filename = if args.output.ends_with(".wav") {
            args.output
                .replace(".wav", &format!("_{:03}.wav", segment_num))
        } else {
            format!("{}_{:03}.wav", args.output, segment_num)
        };
        save_output_audio(&audio_output, &output_filename, device, &args.models_dir)?;

        print_results(
            &finnish_text,
            &portuguese_text,
            &output_filename,
            args.show_original,
        );

        println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    }

    println!("✅ All {} segments processed successfully!", segment_num);
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Handle model download mode
    if args.download_models {
        return download_models(&args.models_dir);
    }

    println!("===== Finnish to Portuguese Speech Translation =====\n");

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

    println!("Using device: {:?}\n", device);

    // Handle segmented recording mode
    if args.segmented {
        if !args.microphone && args.input.is_none() {
            eprintln!("⚠️  Segmented mode requires --microphone flag");
            anyhow::bail!("Use --microphone --segmented for segmented recording mode");
        }
        return process_segmented_recording(&args, &device);
    }

    // Original single-file mode
    // Get Finnish text
    let finnish_text = if args.test_mode {
        println!("Running in test mode with text: {}", args.test_text);
        args.test_text.clone()
    } else {
        let audio_samples = get_audio_samples(&args)?;
        transcribe_audio(
            &audio_samples,
            &device,
            args.show_original,
            &args.models_dir,
        )?
    };

    // Translate and synthesize
    let portuguese_text = translate_text(&finnish_text, &device, &args.models_dir)?;
    let audio_output = synthesize_speech(&portuguese_text, &device, &args.models_dir)?;
    save_output_audio(&audio_output, &args.output, &device, &args.models_dir)?;

    print_results(
        &finnish_text,
        &portuguese_text,
        &args.output,
        args.show_original,
    );

    // Evaluate speech quality if requested (test mode only)
    if args.evaluate && args.test_mode {
        println!("\n🔍 Evaluating synthesized speech quality...");
        let mut evaluator = speech_evaluator::PortugueseEvaluator::new(device.clone())?;
        let result =
            evaluator.evaluate_quality(&portuguese_text, &audio_output, &args.models_dir)?;
        result.print_report();
    } else if args.evaluate && !args.test_mode {
        println!("\n⚠️  Note: --evaluate flag is only available in --test-mode");
    }

    Ok(())
}
