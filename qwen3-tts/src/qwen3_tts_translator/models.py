"""
Model download and management utilities.
"""

import sys
from pathlib import Path
from typing import Optional

from huggingface_hub import snapshot_download


# Model configurations
MODELS = {
    "whisper": {
        "id": "Finnish-NLP/whisper-tiny-finnish",
        "files": ["config.json", "model.safetensors", "tokenizer.json"],
    },
    "fi_en_translation": {
        "id": "Helsinki-NLP/opus-tatoeba-fi-en",
        "files": ["config.json", "model.safetensors", "source.spm", "target.spm", "vocab.json"],
    },
    "en_pt_translation": {
        "id": "Helsinki-NLP/opus-mt-tc-big-en-pt",
        "files": ["config.json", "model.safetensors", "source.spm", "target.spm", "vocab.json"],
    },
    "qwen_tts_tokenizer": {
        "id": "Qwen/Qwen3-TTS-Tokenizer-12Hz",
        "files": ["config.json"],
    },
    "qwen_tts": {
        "id": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        "files": ["config.json", "model.safetensors"],
    },
}


def get_model_path(model_id: str, models_dir: Optional[Path] = None) -> Path:
    """
    Get local path for a model.

    Args:
        model_id: HuggingFace model ID (e.g., "Finnish-NLP/whisper-tiny-finnish")
        models_dir: Base models directory (defaults to ./models)

    Returns:
        Path to model directory
    """
    if models_dir is None:
        models_dir = Path(__file__).parent.parent.parent / "models"

    # Replace / with -- in model ID
    model_name = model_id.replace("/", "--")
    return models_dir / model_name


def is_model_downloaded(model_id: str, models_dir: Optional[Path] = None) -> bool:
    """
    Check if a model is already downloaded.

    Args:
        model_id: HuggingFace model ID
        models_dir: Base models directory

    Returns:
        True if model exists locally
    """
    model_path = get_model_path(model_id, models_dir)
    return model_path.exists() and any(model_path.iterdir())


def download_model(
    model_id: str,
    models_dir: Optional[Path] = None,
    force: bool = False,
) -> Path:
    """
    Download a model from HuggingFace Hub.

    Args:
        model_id: HuggingFace model ID
        models_dir: Base models directory
        force: Force re-download even if exists

    Returns:
        Path to downloaded model
    """
    if models_dir is None:
        models_dir = Path(__file__).parent.parent.parent / "models"

    model_path = get_model_path(model_id, models_dir)

    # Check if already downloaded
    if not force and is_model_downloaded(model_id, models_dir):
        print(f"✓ Model already downloaded: {model_id}", file=sys.stderr)
        return model_path

    print(f"📥 Downloading {model_id}...", file=sys.stderr)

    # Create models directory
    models_dir.mkdir(parents=True, exist_ok=True)

    # Download from HuggingFace
    downloaded_path = snapshot_download(
        repo_id=model_id,
        local_dir=str(model_path),
        local_dir_use_symlinks=False,
    )

    print(f"✓ Downloaded to {model_path}", file=sys.stderr)
    return Path(downloaded_path)


def download_all_models(
    models_dir: Optional[Path] = None,
    force: bool = False,
):
    """
    Download all required models.

    Args:
        models_dir: Base models directory
        force: Force re-download even if exists
    """
    print("🤖 Downloading all models...", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    for name, config in MODELS.items():
        model_id = config["id"]
        print(f"\n[{name}] {model_id}", file=sys.stderr)
        try:
            download_model(model_id, models_dir, force)
        except Exception as e:
            print(f"❌ Failed to download {model_id}: {e}", file=sys.stderr)
            raise

    print("\n✅ All models downloaded successfully!", file=sys.stderr)


def list_models(models_dir: Optional[Path] = None):
    """
    List all configured models and their download status.

    Args:
        models_dir: Base models directory
    """
    if models_dir is None:
        models_dir = Path(__file__).parent.parent.parent / "models"

    print("📦 Model Status", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    for name, config in MODELS.items():
        model_id = config["id"]
        downloaded = is_model_downloaded(model_id, models_dir)
        status = "✓ Downloaded" if downloaded else "✗ Not downloaded"

        print(f"\n{name}:", file=sys.stderr)
        print(f"  ID: {model_id}", file=sys.stderr)
        print(f"  Status: {status}", file=sys.stderr)

        if downloaded:
            path = get_model_path(model_id, models_dir)
            print(f"  Path: {path}", file=sys.stderr)
