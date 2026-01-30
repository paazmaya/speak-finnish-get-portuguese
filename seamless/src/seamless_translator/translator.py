"""
Core translation functionality using SeamlessM4T v2.
"""

import sys
import time
import warnings
import torch
import soundfile as sf
from typing import Optional, Tuple
from transformers import SeamlessM4Tv2Model, AutoProcessor

# Suppress the layer_idx warning from transformers
warnings.filterwarnings(
    "ignore",
    message=".*layer_idx.*",
    category=FutureWarning,
)


class SeamlessTranslator:
    """
    Speech-to-speech translator using SeamlessM4T v2.

    Supports 100+ languages with 8-bit quantization for efficient inference.
    """

    def __init__(
        self,
        model_name: str = "facebook/seamless-m4t-v2-large",
        device: Optional[str] = None,
        load_in_8bit: bool = False,
    ):
        """
        Initialize the translator.

        Args:
            model_name: HuggingFace model identifier
            device: Device to use (cuda/cpu/mps). Auto-detected if None.
            load_in_8bit: Whether to use 8-bit quantization
        """
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = device
        self.model_name = model_name

        print(f"🖥️  Device: {device}", file=sys.stderr)
        print(f"📦 Model: {model_name}", file=sys.stderr)
        print("⚙️  Loading model...", file=sys.stderr)

        # Load processor from the same model we're using (local only)
        self.processor = AutoProcessor.from_pretrained(
            model_name, local_files_only=True
        )

        # Load model with optional 8-bit quantization (local only)
        model_kwargs = {
            "device_map": device,
            "local_files_only": True,
        }

        if load_in_8bit and device != "mps":  # MPS doesn't support bitsandbytes yet
            model_kwargs["load_in_8bit"] = True
            print("💾 Using 8-bit quantization", file=sys.stderr)
        else:
            if device == "mps":
                print(
                    "⚠️  MPS doesn't support 8-bit quantization - loading full model (slower)",
                    file=sys.stderr,
                )
            print("💾 Loading full precision model", file=sys.stderr)

        self.model = SeamlessM4Tv2Model.from_pretrained(model_name, **model_kwargs)

        self.sample_rate = self.model.config.sampling_rate

        # Get model size info
        param_count = sum(p.numel() for p in self.model.parameters())
        param_size_mb = (
            sum(p.numel() * p.element_size() for p in self.model.parameters()) / 1024**2
        )

        print("✓ Model loaded successfully", file=sys.stderr)
        print(f"  Sample rate: {self.sample_rate} Hz", file=sys.stderr)
        print(
            f"  Parameters: {param_count:,} ({param_size_mb:.1f} MB)", file=sys.stderr
        )

    def load_audio(self, audio_path: str) -> Tuple[torch.Tensor, int]:
        """
        Load and preprocess audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            Tuple of (audio_tensor, sample_rate)
        """
        # Load audio using soundfile to avoid torchcodec dependency
        audio_data, orig_freq = sf.read(audio_path, dtype="float32")

        # Convert to torch tensor
        audio = torch.from_numpy(audio_data)
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)  # Add channel dimension: [1, samples]
        else:
            audio = audio.t()  # Transpose to [channels, samples]

        # Resample to 16kHz (required by SeamlessM4T)
        if orig_freq != 16000:
            import torch.nn.functional as F

            target_length = int(audio.shape[1] * 16000 / orig_freq)
            audio = F.interpolate(
                audio.unsqueeze(0), size=target_length, mode="linear", align_corners=False
            ).squeeze(0)

        # Convert stereo to mono if needed
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)

        return audio, 16000

    def translate(
        self,
        input_audio_path: str,
        output_audio_path: str,
        src_lang: str = "fin",
        tgt_lang: str = "por",
        verbose: bool = True,
    ) -> None:
        """
        Translate speech from source to target language.

        Args:
            input_audio_path: Path to input audio file
            output_audio_path: Path to save output audio
            src_lang: Source language code (e.g., 'fin' for Finnish)
            tgt_lang: Target language code (e.g., 'por' for Portuguese)
            verbose: Whether to print progress messages
        """
        total_start = time.time()

        if verbose:
            print(f"📥 Loading audio: {input_audio_path}", file=sys.stderr)
            print(f"  Source language: {src_lang}", file=sys.stderr)
            print(f"  Target language: {tgt_lang}", file=sys.stderr)

        # Load and preprocess audio
        load_start = time.time()
        audio, sr = self.load_audio(input_audio_path)
        load_time = time.time() - load_start

        if verbose:
            duration = audio.shape[1] / sr
            print(
                f"✓ Audio loaded: {duration:.2f} seconds at {sr} Hz (took {load_time:.2f}s)",
                file=sys.stderr,
            )

        # Prepare audio input for speech-to-speech translation
        # Note: SeamlessM4T auto-detects source language from audio
        prep_start = time.time()
        audio_inputs = self.processor(
            audio=audio.squeeze().numpy(),
            return_tensors="pt",
            sampling_rate=sr,
        )

        # Move inputs to device
        audio_inputs = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in audio_inputs.items()
        }
        prep_time = time.time() - prep_start

        if verbose:
            print(
                f"🎯 Processing audio with model... (preprocessing took {prep_time:.2f}s)",
                file=sys.stderr,
            )
            print(f"🔊 Generating {tgt_lang} speech...", file=sys.stderr)

        # Generate speech output with text
        gen_start = time.time()
        with torch.no_grad():
            # Generate speech-to-speech translation
            # First, get text translation to verify it's actually translating
            try:
                # Try to get text translation first for debugging
                text_output = self.model.generate(
                    **audio_inputs,
                    tgt_lang=tgt_lang,
                    generate_speech=False,  # Text-only generation
                )
                if verbose and hasattr(text_output, 'sequences'):
                    # Decode the text
                    translated_text = self.processor.decode(
                        text_output.sequences[0].tolist(),
                        skip_special_tokens=True
                    )
                    print(f"📝 Translated text: {translated_text}", file=sys.stderr)
            except Exception as e:
                if verbose:
                    print(f"⚠️  Could not extract text: {e}", file=sys.stderr)
            
            # Now generate the speech
            audio_array = (
                self.model.generate(
                    **audio_inputs,
                    tgt_lang=tgt_lang,
                )[0]
                .cpu()
                .squeeze()
            )
        
        gen_time = time.time() - gen_start

        if verbose:
            print(f"✓ Generation complete (took {gen_time:.2f}s)", file=sys.stderr)

        # Save output audio
        if verbose:
            print("💾 Saving translated audio...", file=sys.stderr)

        save_start = time.time()
        # Use soundfile to save (avoids torchcodec dependency)
        # Convert to float32 (soundfile doesn't support float16)
        audio_np = audio_array.cpu().numpy().astype("float32")
        sf.write(
            output_audio_path,
            audio_np,
            samplerate=self.sample_rate,
            subtype="PCM_16",
        )
        save_time = time.time() - save_start
        total_time = time.time() - total_start

        if verbose:
            duration = audio_array.shape[0] / self.sample_rate
            print(f"✓ Saved to {output_audio_path}", file=sys.stderr)
            print(f"  Output duration: {duration:.2f} seconds", file=sys.stderr)
            print(f"  Sample rate: {self.sample_rate} Hz", file=sys.stderr)
            print(
                f"⏱️  Timing: load={load_time:.2f}s, prep={prep_time:.2f}s, gen={gen_time:.2f}s, save={save_time:.2f}s",
                file=sys.stderr,
            )
            print(f"⏱️  Total: {total_time:.2f}s", file=sys.stderr)

    @staticmethod
    def get_supported_languages() -> dict:
        """
        Get information about supported languages.

        Returns:
            Dictionary with language information
        """
        return {
            "note": "SeamlessM4T v2 supports 100+ languages",
            "common_codes": {
                "fin": "Finnish",
                "por": "Portuguese",
                "eng": "English",
                "spa": "Spanish",
                "fra": "French",
                "deu": "German",
                "ita": "Italian",
                "jpn": "Japanese",
                "kor": "Korean",
                "zho": "Chinese",
                "rus": "Russian",
                "ara": "Arabic",
                "hin": "Hindi",
            },
            "docs": "https://huggingface.co/facebook/seamless-m4t-v2-large#supported-languages",
        }
