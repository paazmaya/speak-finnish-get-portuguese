mod config;
mod qwen3_speech_tokenizer;
mod qwen3_tts_model;
mod text_to_speech;
mod translation;

use anyhow::Result;
use candle_core::Device;
use clap::{Parser, ValueEnum};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about = "Finnish to Portuguese speech translation using Qwen3-TTS", long_about = None)]
struct Args {
    /// Input text in Finnish (overrides test mode default)
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

    // Get Finnish input text
    let finnish_text = if let Some(t) = args.text {
        println!("Finnish input text: {}\n", t);
        t
    } else if args.test_mode {
        let default_text = "Hei, miten voit tänään?";
        println!("Running in test mode with Finnish text: {}\n", default_text);
        default_text.to_string()
    } else {
        println!("Please provide Finnish text using --text \"Your Finnish text here\" or use --test-mode");
        println!("Example: cargo run --release -- --text \"Hei, kuinka voit?\"");
        println!("Example: cargo run --release -- --test-mode");
        anyhow::bail!("No input text provided");
    };

    // Translate Finnish to Portuguese
    println!("Step 1: Translating from Finnish to Portuguese...");
    let mut translator = translation::Translator::new(device.clone(), &args.translation_models_dir)?;
    let portuguese_text = translator.translate(&finnish_text)?;
    println!("Portuguese translation: {}\n", portuguese_text);

    // Synthesize Portuguese speech
    println!("Step 2: Synthesizing Portuguese speech...");
    let mut tts = text_to_speech::PortugueseTTS::new(device.clone(), &args.models_dir)?;
    let audio_output = tts.synthesize(&portuguese_text)?;
    
    // Save audio
    println!("\nStep 3: Saving audio...");
    tts.save_wav(&audio_output, &args.output)?;

    println!("\n===== Translation Pipeline Complete =====");
    println!("Finnish: {}", finnish_text);
    println!("Portuguese: {}", portuguese_text);
    println!("Audio saved to: {}", args.output);

    Ok(())
}
