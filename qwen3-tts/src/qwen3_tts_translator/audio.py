"""
Audio recording from microphone with keyboard controls.
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd
import soundfile as sf


def list_devices():
    """List all available audio devices."""
    return sd.query_devices()


def get_default_input():
    """Get default input device."""
    return sd.query_devices(kind="input")


def record_audio(duration: float, sample_rate: int = 16000) -> np.ndarray:
    """
    Record audio from microphone.

    Args:
        duration: Recording duration in seconds
        sample_rate: Sample rate in Hz (default 16kHz for Whisper)

    Returns:
        Audio data as float32 numpy array
    """
    print(f"🎤 Recording for {duration:.1f} seconds...", file=sys.stderr)

    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
    )
    sd.wait()

    print("✓ Recording complete", file=sys.stderr)
    return audio.flatten()


def save_audio(audio: np.ndarray, path: str, sample_rate: int = 16000):
    """
    Save audio to WAV file.

    Args:
        audio: Audio data as numpy array
        path: Output file path
        sample_rate: Sample rate in Hz
    """
    sf.write(path, audio, sample_rate)
    print(f"💾 Saved audio to {path}", file=sys.stderr)


def load_audio(path: str) -> tuple[np.ndarray, int]:
    """
    Load audio from file.

    Args:
        path: Input audio file path

    Returns:
        Tuple of (audio_data, sample_rate)
    """
    audio, sr = sf.read(path, dtype="float32")

    # Convert stereo to mono if needed
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    return audio, sr


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    Resample audio to target sample rate.

    Args:
        audio: Input audio data
        orig_sr: Original sample rate
        target_sr: Target sample rate

    Returns:
        Resampled audio
    """
    if orig_sr == target_sr:
        return audio

    # Simple linear interpolation resampling
    duration = len(audio) / orig_sr
    num_samples = int(duration * target_sr)
    resampled = np.interp(
        np.linspace(0, len(audio) - 1, num_samples),
        np.arange(len(audio)),
        audio,
    )
    return resampled.astype(np.float32)


class AudioRecorder:
    """
    Real-time audio recorder with callback support.
    """

    def __init__(self, sample_rate: int = 16000):
        """
        Initialize audio recorder.

        Args:
            sample_rate: Recording sample rate
        """
        self.sample_rate = sample_rate
        self.frames = []
        self.is_recording = False
        self.stream: Optional[sd.InputStream] = None

    def _callback(self, indata, frames, time_info, status):
        """Audio input callback."""
        if status:
            print(f"Recording status: {status}", file=sys.stderr)
        if self.is_recording:
            self.frames.append(indata.copy())

    def start(self):
        """Start recording."""
        if self.is_recording:
            return

        self.is_recording = True
        self.frames = []

        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            callback=self._callback,
        )
        self.stream.start()

    def stop(self) -> np.ndarray:
        """
        Stop recording and return audio data.

        Returns:
            Recorded audio as numpy array
        """
        if not self.is_recording:
            return np.array([], dtype=np.float32)

        self.is_recording = False

        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        if not self.frames:
            return np.array([], dtype=np.float32)

        return np.concatenate(self.frames, axis=0).flatten()

    def get_duration(self) -> float:
        """Get duration of currently recorded audio in seconds."""
        if not self.frames:
            return 0.0
        total_samples = sum(len(f) for f in self.frames)
        return total_samples / self.sample_rate
