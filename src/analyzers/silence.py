"""Silence analyzer — detects leading and trailing silence.

Computes RMS energy in short frames and identifies silence regions
at the head and tail of the track.
"""

import numpy as np
import soundfile as sf
from pathlib import Path

from src.models.report import SilenceReport


def compute_rms_frames(
    samples: np.ndarray,
    sample_rate: int,
    frame_ms: float = 10.0,
) -> np.ndarray:
    """Compute RMS energy per frame.

    Args:
        samples: Audio as float32 array. If 2D, averaged to mono first.
        sample_rate: Sample rate in Hz.
        frame_ms: Frame length in milliseconds.

    Returns:
        1D array of RMS values per frame.
    """
    if samples.ndim == 2:
        samples = samples.mean(axis=1)

    frame_size = int(sample_rate * frame_ms / 1000.0)
    if frame_size == 0:
        return np.array([])

    num_frames = len(samples) // frame_size
    if num_frames == 0:
        return np.array([])

    # Reshape into frames and compute RMS per frame
    trimmed = samples[: num_frames * frame_size]
    frames = trimmed.reshape(num_frames, frame_size)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))

    return rms


def find_leading_silence(
    rms_frames: np.ndarray,
    frame_ms: float,
    threshold_db: float = -60.0,
) -> float:
    """Find duration of leading silence in milliseconds.

    Args:
        rms_frames: RMS energy per frame.
        frame_ms: Duration of each frame in ms.
        threshold_db: RMS below this (in dBFS) counts as silence.

    Returns:
        Leading silence duration in milliseconds.
    """
    if len(rms_frames) == 0:
        return 0.0

    threshold_linear = 10 ** (threshold_db / 20.0)

    for i, rms in enumerate(rms_frames):
        if rms > threshold_linear:
            return i * frame_ms

    # Entire track is silent
    return len(rms_frames) * frame_ms


def find_trailing_silence(
    rms_frames: np.ndarray,
    frame_ms: float,
    threshold_db: float = -60.0,
) -> float:
    """Find duration of trailing silence in milliseconds.

    Args:
        rms_frames: RMS energy per frame.
        frame_ms: Duration of each frame in ms.
        threshold_db: RMS below this (in dBFS) counts as silence.

    Returns:
        Trailing silence duration in milliseconds.
    """
    if len(rms_frames) == 0:
        return 0.0

    threshold_linear = 10 ** (threshold_db / 20.0)

    for i in range(len(rms_frames) - 1, -1, -1):
        if rms_frames[i] > threshold_linear:
            remaining_frames = len(rms_frames) - 1 - i
            return remaining_frames * frame_ms

    # Entire track is silent
    return len(rms_frames) * frame_ms


def analyze_silence(
    path: str | Path,
    threshold_db: float = -60.0,
    min_silence_ms: float = 100.0,
) -> SilenceReport:
    """Run silence analysis on an audio file.

    Args:
        path: Path to audio file.
        threshold_db: RMS below this counts as silence (default: -60 dBFS).
        min_silence_ms: Silence shorter than this is ignored (default: 100ms).

    Returns:
        SilenceReport with leading/trailing silence measurements.

    Raises:
        FileNotFoundError: If file doesn't exist.
        RuntimeError: If file can't be decoded.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    try:
        samples, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception as e:
        raise RuntimeError(f"Failed to decode {path.name}: {e}") from e

    frame_ms = 10.0
    rms_frames = compute_rms_frames(samples, sample_rate, frame_ms)

    leading = find_leading_silence(rms_frames, frame_ms, threshold_db)
    trailing = find_trailing_silence(rms_frames, frame_ms, threshold_db)

    return SilenceReport(
        leading_silence_ms=round(leading, 1),
        trailing_silence_ms=round(trailing, 1),
        leading_trimmed=leading > min_silence_ms,
        trailing_trimmed=trailing > min_silence_ms,
    )
