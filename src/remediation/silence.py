"""Silence remediation — trim leading/trailing silence with fade.

Removes dead air at the head and tail of a track, applying a short
fade to avoid clicks at the edit points.
"""

import numpy as np
import soundfile as sf
from pathlib import Path


def apply_fade(samples: np.ndarray, sample_rate: int, fade_ms: float = 10.0) -> np.ndarray:
    """Apply fade-in at start and fade-out at end.

    A 10ms linear fade prevents clicks at edit points. This is
    imperceptible to listeners but eliminates discontinuity pops.

    Args:
        samples: Audio as float32 array, shape (num_samples,) or (num_samples, channels).
        sample_rate: Sample rate in Hz.
        fade_ms: Fade duration in milliseconds.

    Returns:
        Samples with fades applied.
    """
    fade_samples = int(sample_rate * fade_ms / 1000.0)
    fade_samples = min(fade_samples, len(samples) // 2)  # don't exceed half the track

    if fade_samples == 0:
        return samples

    result = samples.copy()
    ramp = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)

    if result.ndim == 1:
        result[:fade_samples] *= ramp
        result[-fade_samples:] *= ramp[::-1]
    else:
        result[:fade_samples] *= ramp[:, np.newaxis]
        result[-fade_samples:] *= ramp[::-1, np.newaxis]

    return result


def trim_silence(
    input_path: str | Path,
    output_path: str | Path,
    threshold_db: float = -60.0,
    min_silence_ms: float = 100.0,
    fade_ms: float = 10.0,
) -> Path:
    """Trim leading/trailing silence and apply fades.

    The original file is never modified.

    Args:
        input_path: Path to source audio file.
        output_path: Path for trimmed output (WAV 24-bit).
        threshold_db: RMS below this counts as silence.
        min_silence_ms: Only trim if silence exceeds this duration.
        fade_ms: Fade duration at edit points.

    Returns:
        Path to the trimmed file.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    samples, sample_rate = sf.read(str(input_path), dtype="float32", always_2d=True)

    threshold_linear = 10 ** (threshold_db / 20.0)
    frame_size = int(sample_rate * 10.0 / 1000.0)  # 10ms frames

    # Find first non-silent frame
    mono = samples.mean(axis=1)
    num_frames = len(mono) // frame_size
    frames = mono[: num_frames * frame_size].reshape(num_frames, frame_size)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))

    # Leading edge
    start_frame = 0
    for i, val in enumerate(rms):
        if val > threshold_linear:
            start_frame = i
            break

    # Trailing edge
    end_frame = num_frames
    for i in range(num_frames - 1, -1, -1):
        if rms[i] > threshold_linear:
            end_frame = i + 1
            break

    start_sample = start_frame * frame_size
    end_sample = min(end_frame * frame_size, len(samples))

    # Only trim if silence exceeds threshold
    leading_ms = start_frame * 10.0
    trailing_ms = (num_frames - end_frame) * 10.0

    if leading_ms <= min_silence_ms:
        start_sample = 0
    if trailing_ms <= min_silence_ms:
        end_sample = len(samples)

    trimmed = samples[start_sample:end_sample]

    # Apply fades at edit points
    if len(trimmed) > 0:
        trimmed = apply_fade(trimmed, sample_rate, fade_ms)

    sf.write(str(output_path), trimmed, sample_rate, subtype="PCM_24")
    return output_path
