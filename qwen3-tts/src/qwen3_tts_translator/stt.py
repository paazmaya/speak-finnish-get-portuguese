"""
Finnish speech-to-text using Whisper model.
"""

import sys
import torch
from pathlib import Path
from typing import Optional

from transformers import WhisperProcessor, WhisperForConditionalGeneration
import numpy as np

from .models import get_model_path, is_model_downloaded


class FinnishSTT:
    """
    Finnish speech-to-text transcriber using Whisper.
    """

    def __init__(
        self,
        model_name: str = "Finnish-NLP/whisper-tiny-finnish",
        device: Optional[str] = None,
    ):
        """
        Initialize Finnish STT model.

        Args:
            model_name: HuggingFace model identifier
            device: Device to use (cuda/cpu/mps). Auto-detected if None.
        """
        self.device = self._select_device(device)
        self.model_name = model_name

        print(f"🖥️  Device: {self.device}", file=sys.stderr)
        print(f"📦 STT Model: {model_name}", file=sys.stderr)
        print("⚙️  Loading Whisper model...", file=sys.stderr)

        # Use local model if available
        if is_model_downloaded(model_name):
            model_path = str(get_model_path(model_name))
            self.processor = WhisperProcessor.from_pretrained(
                model_path, local_files_only=True
            )
            self.model = WhisperForConditionalGeneration.from_pretrained(
                model_path, local_files_only=True
            ).to(self.device)
        else:
            self.processor = WhisperProcessor.from_pretrained(
                model_name, local_files_only=False
            )
            self.model = WhisperForConditionalGeneration.from_pretrained(
                model_name, local_files_only=False
            ).to(self.device)

        print("✓ Whisper model loaded", file=sys.stderr)

    def _select_device(self, device: Optional[str]) -> str:
        """Select the best available device."""
        if device:
            return device

        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Transcribe Finnish audio to text.

        Args:
            audio: Audio data as numpy array
            sample_rate: Sample rate (should be 16kHz for Whisper)

        Returns:
            Transcribed text in Finnish
        """
        if sample_rate != 16000:
            raise ValueError("Whisper requires 16kHz audio")

        # Process audio
        inputs = self.processor(
            audio, sampling_rate=sample_rate, return_tensors="pt"
        )
        input_features = inputs.input_features.to(self.device)

        # Generate transcription
        with torch.no_grad():
            predicted_ids = self.model.generate(input_features)

        # Decode transcription
        transcription = self.processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0]

        return transcription.strip()

    def transcribe_file(self, audio_path: str) -> str:
        """
        Transcribe audio from file.

        Args:
            audio_path: Path to audio file

        Returns:
            Transcribed text
        """
        import soundfile as sf

        audio, sr = sf.read(audio_path, dtype="float32")

        # Convert to mono if stereo
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Resample if needed
        if sr != 16000:
            from .audio import resample_audio

            audio = resample_audio(audio, sr, 16000)

        return self.transcribe(audio, sample_rate=16000)
