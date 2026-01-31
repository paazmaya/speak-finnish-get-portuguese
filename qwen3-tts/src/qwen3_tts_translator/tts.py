"""
Portuguese text-to-speech using official Qwen3-TTS.
"""

import sys
import torch
import soundfile as sf
from typing import Optional

from qwen_tts import Qwen3TTSModel

from .models import get_model_path, is_model_downloaded


class PortugueseTTS:
    """
    Portuguese text-to-speech synthesizer using Qwen3-TTS.

    Uses the official qwen-tts package for high-quality Brazilian
    Portuguese speech synthesis.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        device: Optional[str] = None,
        speaker: str = "Aiden",
        instruction: str = "Speak in a clear and natural tone",
    ):
        """
        Initialize Qwen3-TTS model.

        Args:
            model_name: HuggingFace model identifier
            device: Device to use (cuda/cpu/mps)
            speaker: Voice speaker name (Ryan, Aiden, etc.)
            instruction: Voice style instruction
        """
        self.device = self._select_device(device)
        self.speaker = speaker
        self.instruction = instruction
        self.model_name = model_name

        print(f"🖥️  Device: {self.device}", file=sys.stderr)
        print(f"📦 TTS Model: {model_name}", file=sys.stderr)
        print(f"🎤 Speaker: {speaker}", file=sys.stderr)
        
        # Check if model is available locally
        model_path = get_model_path(model_name)
        if model_path.exists():
            self.model_path = str(model_path)
            print(f"✓ Using local model: {model_path}", file=sys.stderr)
        else:
            self.model_path = model_name
            print(f"⬇️  Will download from HuggingFace: {model_name}", file=sys.stderr)

        # Lazy load model
        self._model = None
        self._sample_rate = 24000  # Qwen3-TTS default sample rate

    def _select_device(self, device: Optional[str]) -> str:
        """Select best available device."""
        if device:
            return device

        if torch.cuda.is_available():
            return "cuda:0"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load_model(self):
        """Lazy load the Qwen3-TTS model."""
        if self._model is not None:
            return

        # Determine dtype and attention implementation
        dtype = torch.bfloat16 if self.device != "cpu" else torch.float32
        attn_impl = "flash_attention_2" if self.device.startswith("cuda") else None

        print(f"⏳ Loading Qwen3-TTS model...", file=sys.stderr)
        
        self._model = Qwen3TTSModel.from_pretrained(
            self.model_path,
            device_map=self.device,
            dtype=dtype,
            attn_implementation=attn_impl,
        )

        print(f"✓ Qwen3-TTS model loaded on {self.device}", file=sys.stderr)

    def synthesize(
        self,
        text: str,
        language: str = "Portuguese",
        speaker: Optional[str] = None,
        instruction: Optional[str] = None,
    ) -> tuple[list, int]:
        """
        Synthesize speech from Portuguese text.

        Args:
            text: Portuguese text to synthesize
            language: Language name (default: Portuguese)
            speaker: Speaker voice (uses instance default if None)
            instruction: Voice instruction (uses instance default if None)

        Returns:
            Tuple of (waveforms, sample_rate)
        """
        self._load_model()

        # Use instance defaults if not provided
        speaker = speaker or self.speaker
        instruction = instruction or self.instruction

        # Generate speech using official Qwen3-TTS API
        wavs, sr = self._model.generate_custom_voice(
            text=text,
            language=language,
            speaker=speaker,
            instruct=instruction,
            max_new_tokens=2048,
        )

        self._sample_rate = sr
        return wavs, sr

    def synthesize_to_file(
        self,
        text: str,
        output_path: str,
        language: str = "Portuguese",
        speaker: Optional[str] = None,
        instruction: Optional[str] = None,
    ):
        """
        Synthesize speech and save to file.

        Args:
            text: Portuguese text to synthesize
            output_path: Output WAV file path
            language: Language name
            speaker: Speaker voice
            instruction: Voice instruction
        """
        wavs, sr = self.synthesize(text, language, speaker, instruction)

        # Save first waveform (index 0)
        sf.write(output_path, wavs[0], sr)
        print(f"💾 Saved audio to {output_path}", file=sys.stderr)

    @staticmethod
    def list_speakers() -> dict:
        """
        Get available speakers and their descriptions.

        Returns:
            Dictionary of speaker names and descriptions
        """
        return {
            "Vivian": "Bright young female voice (Chinese)",
            "Serena": "Warm, gentle young female voice (Chinese)",
            "Uncle_Fu": "Seasoned male voice, mellow timbre (Chinese)",
            "Dylan": "Youthful Beijing male voice (Chinese - Beijing)",
            "Eric": "Lively Chengdu male voice (Chinese - Sichuan)",
            "Ryan": "Dynamic male voice with rhythm (English/Portuguese)",
            "Aiden": "Sunny American male voice (English/Portuguese)",
            "Ono_Anna": "Playful Japanese female voice (Japanese)",
            "Sohee": "Warm Korean female voice (Korean)",
        }
