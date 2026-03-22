"""Shared test fixtures — synthetic audio generation.

All tests use generated audio so they're deterministic, fast,
and don't require real music files.
"""

import numpy as np
import soundfile as sf
import pytest
from pathlib import Path


@pytest.fixture
def tmp_audio_dir(tmp_path):
    """Temporary directory for test audio files."""
    return tmp_path


def generate_sine(
    frequency: float = 440.0,
    duration: float = 5.0,
    sample_rate: int = 44100,
    amplitude: float = 0.5,
    channels: int = 1,
) -> tuple[np.ndarray, int]:
    """Generate a sine wave as float32 samples.

    Args:
        frequency: Frequency in Hz.
        duration: Duration in seconds.
        sample_rate: Sample rate in Hz.
        amplitude: Peak amplitude (0.0 to 1.0).
        channels: Number of channels.

    Returns:
        Tuple of (samples as 2D array, sample_rate).
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False, dtype=np.float32)
    mono = (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.float32)

    if channels == 1:
        return mono[:, np.newaxis], sample_rate
    else:
        return np.column_stack([mono] * channels), sample_rate


def write_test_wav(
    path: Path,
    frequency: float = 440.0,
    duration: float = 5.0,
    sample_rate: int = 44100,
    amplitude: float = 0.5,
    channels: int = 1,
) -> Path:
    """Write a synthetic WAV file for testing."""
    samples, sr = generate_sine(frequency, duration, sample_rate, amplitude, channels)
    sf.write(str(path), samples, sr, subtype="PCM_24")
    return path


@pytest.fixture
def quiet_track(tmp_audio_dir) -> Path:
    """A quiet track at roughly -20 LUFS."""
    return write_test_wav(tmp_audio_dir / "quiet.wav", amplitude=0.1, duration=5.0)


@pytest.fixture
def loud_track(tmp_audio_dir) -> Path:
    """A loud track at roughly -6 LUFS (well above Spotify's -14 target)."""
    return write_test_wav(tmp_audio_dir / "loud.wav", amplitude=0.9, duration=5.0)


@pytest.fixture
def clipped_track(tmp_audio_dir) -> Path:
    """A track with hard clipping — samples at exactly ±1.0."""
    samples, sr = generate_sine(440.0, 5.0, 44100, amplitude=1.5)
    samples = np.clip(samples, -1.0, 1.0)  # hard clip
    path = tmp_audio_dir / "clipped.wav"
    sf.write(str(path), samples, sr, subtype="PCM_24")
    return path


@pytest.fixture
def normal_track(tmp_audio_dir) -> Path:
    """A track near -14 LUFS — close to Spotify's target."""
    return write_test_wav(tmp_audio_dir / "normal.wav", amplitude=0.30, duration=5.0)


@pytest.fixture
def stereo_track(tmp_audio_dir) -> Path:
    """A stereo track."""
    return write_test_wav(
        tmp_audio_dir / "stereo.wav", amplitude=0.5, duration=5.0, channels=2
    )
