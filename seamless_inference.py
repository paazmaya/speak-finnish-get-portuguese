#!/usr/bin/env python3
"""
Seamless M4T v2 speech-to-speech translation wrapper.
This script provides a simple interface for Finnish to Portuguese translation.
"""

import argparse
import sys
import torch
import torchaudio
from pathlib import Path
from transformers import SeamlessM4Tv2Model, AutoProcessor


def load_model(device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    """Load the 8-bit quantized SeamlessM4T v2 model."""
    print(f"Loading model on {device}...", file=sys.stderr)
    
    model_name = "xun/seamless-m4t-v2-large-8bit-bnb"
    
    # Load processor
    processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
    
    # Load model with 8-bit quantization
    model = SeamlessM4Tv2Model.from_pretrained(
        model_name,
        device_map=device,
        load_in_8bit=True,
    )
    
    print("Model loaded successfully", file=sys.stderr)
    return model, processor


def translate_audio(
    model,
    processor,
    input_audio_path: str,
    output_audio_path: str,
    src_lang: str = "fin",  # Finnish
    tgt_lang: str = "por",  # Portuguese
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Translate audio from source language to target language.
    
    Args:
        model: SeamlessM4T model
        processor: SeamlessM4T processor
        input_audio_path: Path to input audio file
        output_audio_path: Path to save output audio
        src_lang: Source language code (default: fin for Finnish)
        tgt_lang: Target language code (default: por for Portuguese)
        device: Device to use for inference
    """
    print(f"Translating {input_audio_path}...", file=sys.stderr)
    
    # Load and resample audio to 16kHz
    audio, orig_freq = torchaudio.load(input_audio_path)
    audio = torchaudio.functional.resample(
        audio, orig_freq=orig_freq, new_freq=16000
    )
    
    # Convert stereo to mono if needed
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    
    # Prepare audio input
    audio_inputs = processor(
        audios=audio.squeeze().numpy(),
        src_lang=src_lang,
        return_tensors="pt",
        sampling_rate=16000,
    )
    
    # Move inputs to device
    audio_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                   for k, v in audio_inputs.items()}
    
    print(f"Generating Portuguese speech...", file=sys.stderr)
    
    # Generate speech output
    with torch.no_grad():
        audio_array = model.generate(
            **audio_inputs,
            tgt_lang=tgt_lang,
            return_intermediate_token_ids=False,
        )[0].cpu().squeeze()
    
    # Save output audio
    sample_rate = model.config.sampling_rate
    torchaudio.save(
        output_audio_path,
        audio_array.unsqueeze(0),
        sample_rate=sample_rate,
    )
    
    print(f"Saved to {output_audio_path}", file=sys.stderr)
    print(f"Sample rate: {sample_rate} Hz", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Finnish to Portuguese speech translation using SeamlessM4T v2"
    )
    parser.add_argument(
        "input",
        type=str,
        help="Input audio file path (WAV format recommended)",
    )
    parser.add_argument(
        "output",
        type=str,
        help="Output audio file path",
    )
    parser.add_argument(
        "--src-lang",
        type=str,
        default="fin",
        help="Source language code (default: fin)",
    )
    parser.add_argument(
        "--tgt-lang",
        type=str,
        default="por",
        help="Target language code (default: por)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu)",
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    
    # Load model
    model, processor = load_model(device=args.device)
    
    # Translate
    translate_audio(
        model=model,
        processor=processor,
        input_audio_path=args.input,
        output_audio_path=args.output,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        device=args.device,
    )


if __name__ == "__main__":
    main()
