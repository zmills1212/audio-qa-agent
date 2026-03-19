"""Loudness analyzer — ITU-R BS.1770-4 LUFS and true peak measurement.

Measures:
  - Integrated loudness (LUFS) via pyloudnorm
  - True peak (dBTP) via 4x oversampling
  - Loudness range (LRA) via short-term statistics
  - Short-term max LUFS (3-second window)
  - Sample peak (dBFS)
"""

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from pathlib import Path
from scipy.signal import resample_poly

from src.models.report import LoudnessReport


def measure_true_peak(samples: np.ndarray, sample_rate: int) -> float:
    """Measure true peak via 4x oversampling per BS.1770-4.

    Digital audio can have inter-sample peaks that exceed the sample peak.
    Upsampling by 4x and finding the max reveals the actual continuous-time peak.

    Args:
        samples: Audio as 1D or 2D float array. If 2D, shape is (num_samples, channels).
        sample_rate: Sample rate in Hz (unused, but kept for interface consistency).

    Returns:
        True peak in dBTP. Returns -inf for silent audio.
    """
    if samples.ndim == 1:
        samples = samples[:, np.newaxis]

    peak = 0.0
    for ch in range(samples.shape[1]):
        channel = samples[:, ch]
        upsampled = resample_poly(channel, up=4, down=1)
        channel_peak = np.max(np.abs(upsampled))
        peak = max(peak, channel_peak)

    if peak == 0:
        return float("-inf")
    return float(20 * np.log10(peak))


def measure_loudness_range(samples: np.ndarray, sample_rate: int) -> float:
    """Measure loudness range (LRA) in LU.

    LRA captures how dynamic the track is — the spread between soft and loud
    passages. Computed from short-term loudness (3s windows) using the
    10th and 95th percentiles per EBU R128.

    Args:
        samples: Audio as float array. If 1D, treated as mono.
        sample_rate: Sample rate in Hz.

    Returns:
        Loudness range in LU. Returns 0.0 if track is too short.
    """
    meter = pyln.Meter(sample_rate, block_size=3.0)

    if samples.ndim == 1:
        samples = samples[:, np.newaxis]

    # Need at least one full 3-second block
    min_samples = int(3.0 * sample_rate)
    if len(samples) < min_samples:
        return 0.0

    window_samples = int(3.0 * sample_rate)
    hop_samples = int(1.0 * sample_rate)  # 1-second hop
    short_term = []

    for start in range(0, len(samples) - window_samples + 1, hop_samples):
        block = samples[start : start + window_samples]
        loudness = meter.integrated_loudness(block)
        if loudness > -70.0:  # gate silence
            short_term.append(loudness)

    if len(short_term) < 2:
        return 0.0

    low = float(np.percentile(short_term, 10))
    high = float(np.percentile(short_term, 95))
    return high - low


def measure_short_term_max(samples: np.ndarray, sample_rate: int) -> float:
    """Find the loudest 3-second window in the track.

    Args:
        samples: Audio as float array.
        sample_rate: Sample rate in Hz.

    Returns:
        Maximum short-term LUFS. Returns -inf if track is too short.
    """
    meter = pyln.Meter(sample_rate, block_size=3.0)

    if samples.ndim == 1:
        samples = samples[:, np.newaxis]

    min_samples = int(3.0 * sample_rate)
    if len(samples) < min_samples:
        return float("-inf")

    window_samples = int(3.0 * sample_rate)
    hop_samples = int(1.0 * sample_rate)
    max_loudness = float("-inf")

    for start in range(0, len(samples) - window_samples + 1, hop_samples):
        block = samples[start : start + window_samples]
        loudness = meter.integrated_loudness(block)
        if loudness > max_loudness:
            max_loudness = loudness

    return float(max_loudness)


def analyze_loudness(path: str | Path) -> LoudnessReport:
    """Run full loudness analysis on an audio file.

    This is the main entry point. Loads the file once, runs all measurements,
    and returns a structured report.

    Args:
        path: Path to audio file (WAV, FLAC, MP3, etc.).

    Returns:
        LoudnessReport with all measurements.

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

    # Integrated loudness (BS.1770-4)
    meter = pyln.Meter(sample_rate)
    integrated = meter.integrated_loudness(samples)

    # True peak via 4x oversampling
    true_peak = measure_true_peak(samples, sample_rate)

    # Sample peak (no oversampling)
    sample_peak_linear = np.max(np.abs(samples))
    sample_peak_db = (
        float(20 * np.log10(sample_peak_linear)) if sample_peak_linear > 0 else float("-inf")
    )

    # Loudness range
    lra = measure_loudness_range(samples, sample_rate)

    # Short-term max
    st_max = measure_short_term_max(samples, sample_rate)

    return LoudnessReport(
        integrated_lufs=float(integrated),
        true_peak_dbtp=true_peak,
        loudness_range_lu=lra,
        short_term_max_lufs=st_max,
        sample_peak_dbfs=sample_peak_db,
    )
