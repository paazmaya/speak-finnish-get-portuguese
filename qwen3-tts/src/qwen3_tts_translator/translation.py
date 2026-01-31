"""
Text translation from Finnish to Portuguese via English.
"""

import sys
import torch
from typing import Optional

from transformers import MarianMTModel, MarianTokenizer

from .models import get_model_path, is_model_downloaded


class TextTranslator:
    """
    Two-stage translator: Finnish → English → Portuguese.

    Uses Helsinki-NLP Marian models for neural machine translation.
    """

    def __init__(
        self,
        fi_en_model: str = "Helsinki-NLP/opus-tatoeba-fi-en",
        en_pt_model: str = "Helsinki-NLP/opus-mt-tc-big-en-pt",
        device: Optional[str] = None,
    ):
        """
        Initialize translation models.

        Args:
            fi_en_model: Finnish to English model
            en_pt_model: English to Portuguese model
            device: Device to use (cuda/cpu/mps)
        """
        self.device = self._select_device(device)

        print(f"🖥️  Device: {self.device}", file=sys.stderr)
        print("⚙️  Loading translation models...", file=sys.stderr)

        # Load Finnish → English model
        print(f"📦 Loading {fi_en_model}...", file=sys.stderr)
        fi_en_path = str(get_model_path(fi_en_model)) if is_model_downloaded(fi_en_model) else fi_en_model
        self.fi_en_tokenizer = MarianTokenizer.from_pretrained(fi_en_path)
        self.fi_en_model = MarianMTModel.from_pretrained(fi_en_path).to(self.device)

        # Load English → Portuguese model
        print(f"📦 Loading {en_pt_model}...", file=sys.stderr)
        en_pt_path = str(get_model_path(en_pt_model)) if is_model_downloaded(en_pt_model) else en_pt_model
        self.en_pt_tokenizer = MarianTokenizer.from_pretrained(en_pt_path)
        self.en_pt_model = MarianMTModel.from_pretrained(en_pt_path).to(self.device)

        print("✓ Translation models loaded", file=sys.stderr)

    def _select_device(self, device: Optional[str]) -> str:
        """Select best available device."""
        if device:
            return device

        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def translate_fi_to_en(self, text: str) -> str:
        """
        Translate Finnish to English.

        Args:
            text: Finnish text

        Returns:
            English translation
        """
        inputs = self.fi_en_tokenizer(text, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            translated = self.fi_en_model.generate(**inputs)

        return self.fi_en_tokenizer.batch_decode(translated, skip_special_tokens=True)[
            0
        ]

    def translate_en_to_pt(self, text: str) -> str:
        """
        Translate English to Portuguese.

        Args:
            text: English text

        Returns:
            Portuguese translation
        """
        inputs = self.en_pt_tokenizer(text, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            translated = self.en_pt_model.generate(**inputs)

        return self.en_pt_tokenizer.batch_decode(translated, skip_special_tokens=True)[
            0
        ]

    def translate(self, finnish_text: str, verbose: bool = False) -> str:
        """
        Translate Finnish to Portuguese via English.

        Args:
            finnish_text: Finnish input text
            verbose: Print intermediate results

        Returns:
            Portuguese translation
        """
        if verbose:
            print(f"🇫🇮 Finnish: {finnish_text}", file=sys.stderr)

        # Stage 1: Finnish → English
        english_text = self.translate_fi_to_en(finnish_text)

        if verbose:
            print(f"🇬🇧 English: {english_text}", file=sys.stderr)

        # Stage 2: English → Portuguese
        portuguese_text = self.translate_en_to_pt(english_text)

        if verbose:
            print(f"🇵🇹 Portuguese: {portuguese_text}", file=sys.stderr)

        return portuguese_text
