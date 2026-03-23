"""Clipping analyzer — detects hard digital clipping.

Hard clipping occurs when the audio signal exceeds the maximum representable
amplitude and gets clamped flat at ±1.0. This introduces harsh harmonic
distortion. The analyzer finds contiguous regions of clipped samples
and reports severity based on how much of the track is affected.

Three or more consecutive samples at the rail (±1.0) is the standard
indicator of hard clipping vs. a single sample that happens to peak at max.
"""

import numpy as np
import soundfile as sf
from pathlib import Path

from src.models.report import ClippingReport, ClipRegion, Severity


def find_clip_regions(
    samples: np.ndarray,
    channel: int,
    threshold: float = 0.99,
    min_consecutive: int = 3,
) -> list[ClipRegion]:
    """Find contiguous regions where samples are at the rail.

    Args:
        samples: 1D array of samples for a single channel.
        channel: Channel index (for reporting).
        threshold: Amplitude above which a sample is considered clipped.
        min_consecutive: Minimum consecutive clipped samples to count as a region.

    Returns:
        List of ClipRegion objects.
    """
    clipped_mask = np.abs(samples) >= threshold
    regions = []

    in_region = False
    region_start = 0

    for i, is_clipped in enumerate(clipped_mask):
        if is_clipped and not in_region:
            region_start = i
            in_region = True
        elif not is_clipped and in_region:
            length = i - region_start
            if length >= min_consecutive:
                regions.append(ClipRegion(
                    start_sample=region_start,
                    end_sample=i,
                    channel=channel,
                ))
            in_region = False

    # Handle region that extends to end of file
    if in_region:
        length = len(samples) - region_start
        if length >= min_consecutive:
            regions.append(ClipRegion(
                start_sample=region_start,
                end_sample=len(samples),
                channel=channel,
            ))

    return regions


def classify_clipping_severity(clip_percentage: float) -> Severity:
    """Classify clipping severity based on percentage of affected samples.

    Args:
        clip_percentage: Percentage of total samples that are clipped.

    Returns:
        Severity level.
    """
    if clip_percentage == 0:
        return Severity.PASS
    if clip_percentage < 0.01:
        return Severity.WARNING  # trace clipping, barely audible
    return Severity.FAIL  # audible clipping


def analyze_clipping(
    path: str | Path,
    threshold: float = 0.99,
    min_consecutive: int = 3,
) -> ClippingReport:
    """Run clipping analysis on an audio file.

    Args:
        path: Path to audio file.
        threshold: Amplitude at or above which samples are considered clipped.
        min_consecutive: Minimum consecutive samples to count as a clip region.

    Returns:
        ClippingReport with clip regions, counts, and severity.

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

    all_regions = []
    total_clipped_samples = 0

    for ch in range(samples.shape[1]):
        channel_data = samples[:, ch]
        regions = find_clip_regions(channel_data, ch, threshold, min_consecutive)
        all_regions.extend(regions)
        total_clipped_samples += sum(r.length for r in regions)

    total_samples = samples.shape[0] * samples.shape[1]
    clip_pct = (total_clipped_samples / total_samples * 100) if total_samples > 0 else 0.0

    peak_linear = np.max(np.abs(samples))
    peak_db = float(20 * np.log10(peak_linear)) if peak_linear > 0 else float("-inf")

    return ClippingReport(
        clip_count=len(all_regions),
        clipped_samples=total_clipped_samples,
        total_samples=total_samples,
        clip_regions=all_regions,
        peak_dbfs=peak_db,
        severity=classify_clipping_severity(clip_pct),
    )
