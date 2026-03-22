"""Loudness remediation — normalize LUFS and limit true peaks.

Takes a track that failed QA and produces a corrected version.
The original file is never modified.
"""

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from pathlib import Path
from scipy.signal import resample_poly


def normalize_loudness(
    samples: np.ndarray,
    sample_rate: int,
    current_lufs: float,
    target_lufs: float = -14.0,
) -> np.ndarray:
    """Normalize audio to a target integrated loudness.

    Applies a constant gain offset. Does not alter dynamics or tonal character.

    Args:
        samples: Audio as float32 array, shape (num_samples, channels).
        sample_rate: Sample rate in Hz.
        current_lufs: Measured integrated LUFS of the input.
        target_lufs: Desired integrated LUFS.

    Returns:
        Gain-adjusted samples as float32 array.
    """
    delta_db = target_lufs - current_lufs
    gain_linear = 10 ** (delta_db / 20.0)
    return (samples * gain_linear).astype(np.float32)


def limit_true_peak(
    samples: np.ndarray,
    sample_rate: int,
    ceiling_dbtp: float = -1.0,
) -> np.ndarray:
    """Apply a simple true peak limiter.

    Measures the true peak via 4x oversampling. If it exceeds the ceiling,
    attenuates the entire signal by the overshoot amount. This is a
    conservative approach — it preserves dynamics at the cost of overall level.

    A more sophisticated limiter would use lookahead and per-sample gain
    reduction, but that's beyond MVP scope.

    Args:
        samples: Audio as float32 array.
        sample_rate: Sample rate in Hz.
        ceiling_dbtp: Maximum allowed true peak in dBTP.

    Returns:
        Peak-limited samples as float32 array.
    """
    if samples.ndim == 1:
        measure_samples = samples[:, np.newaxis]
    else:
        measure_samples = samples

    # Find true peak across all channels
    peak_linear = 0.0
    for ch in range(measure_samples.shape[1]):
        upsampled = resample_poly(measure_samples[:, ch], up=4, down=1)
        ch_peak = np.max(np.abs(upsampled))
        peak_linear = max(peak_linear, ch_peak)

    if peak_linear == 0:
        return samples

    current_dbtp = 20 * np.log10(peak_linear)

    if current_dbtp <= ceiling_dbtp:
        return samples  # already compliant

    # Attenuate by the overshoot
    reduction_db = ceiling_dbtp - current_dbtp
    gain = 10 ** (reduction_db / 20.0)
    return (samples * gain).astype(np.float32)


def fix_loudness(
    input_path: str | Path,
    output_path: str | Path,
    target_lufs: float = -14.0,
    peak_ceiling_dbtp: float = -1.0,
) -> Path:
    """Full loudness remediation: normalize + peak limit.

    Reads the input, applies corrections, writes a new file.
    The original is never modified.

    Args:
        input_path: Path to source audio file.
        output_path: Path for corrected output (WAV 24-bit).
        target_lufs: Desired integrated LUFS.
        peak_ceiling_dbtp: Maximum true peak in dBTP.

    Returns:
        Path to the corrected file.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    samples, sample_rate = sf.read(str(input_path), dtype="float32", always_2d=True)

    # Step 1: Measure current loudness
    meter = pyln.Meter(sample_rate)
    current_lufs = meter.integrated_loudness(samples)

    # Step 2: Normalize to target
    samples = normalize_loudness(samples, sample_rate, current_lufs, target_lufs)

    # Step 3: Limit true peaks
    samples = limit_true_peak(samples, sample_rate, peak_ceiling_dbtp)

    # Step 4: Write corrected file
    sf.write(str(output_path), samples, sample_rate, subtype="PCM_24")

    return output_path
