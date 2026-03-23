"""Pipeline — orchestrates analyze → decide → fix.

Single entry point for processing a track end-to-end.
"""

import soundfile as sf
from pathlib import Path

from src.models.report import TrackReport, ActionType
from src.analyzers.loudness import analyze_loudness
from src.analyzers.silence import analyze_silence
from src.engine.rules import build_platform_predictions, decide_actions
from src.remediation.loudness import fix_loudness
from src.remediation.silence import trim_silence
from src.platform_specs import PLATFORMS


def strictest_peak_ceiling() -> float:
    """Return the most restrictive true peak limit across all platforms."""
    return min(spec.max_true_peak_dbtp for spec in PLATFORMS.values())


def process_track(
    input_path: str | Path,
    output_dir: str | Path | None = None,
    target_lufs: float = -14.0,
) -> TrackReport:
    """Analyze a track, decide what to fix, and apply corrections.

    This is the main entry point for the entire system.

    Args:
        input_path: Path to the audio file.
        output_dir: Directory for corrected files. If None, uses input file's directory.
        target_lufs: LUFS target for normalization (default: -14.0 for Spotify).

    Returns:
        TrackReport with analysis, predictions, decisions, and path to fixed file.
    """
    input_path = Path(input_path)
    if output_dir is None:
        output_dir = input_path.parent
    output_dir = Path(output_dir)

    # Load file info
    info = sf.info(str(input_path))

    # Analyze
    loudness = analyze_loudness(input_path)
    silence = analyze_silence(input_path)

    # Predict
    predictions = build_platform_predictions(loudness)

    # Decide
    actions = decide_actions(loudness, predictions, target_lufs, silence)

    # Build report
    report = TrackReport(
        source_path=input_path,
        sample_rate=info.samplerate,
        channels=info.channels,
        duration_seconds=info.duration,
        loudness=loudness,
        silence=silence,
        platform_predictions=predictions,
        actions=actions,
    )

    # Fix if needed
    if report.needs_fix:
        stem = input_path.stem
        fixed_name = f"{stem}_fixed.wav"
        fixed_path = output_dir / fixed_name

        # Start with the input file
        current_path = input_path

        # Trim silence first (if needed) — before loudness so LUFS isn't skewed by silence
        if silence.needs_trim:
            trimmed_path = output_dir / f"{stem}_trimmed.wav"
            trim_silence(current_path, trimmed_path)
            current_path = trimmed_path

        # Then fix loudness (if needed)
        lufs_distance = abs(loudness.integrated_lufs - target_lufs)
        has_loudness_issue = lufs_distance > 1.0
        has_peak_issue = any(not p.true_peak_compliant for p in predictions)

        if has_loudness_issue or has_peak_issue:
            fix_loudness(
                input_path=current_path,
                output_path=fixed_path,
                target_lufs=target_lufs,
                peak_ceiling_dbtp=strictest_peak_ceiling(),
            )
            # Clean up intermediate trimmed file
            if current_path != input_path and current_path.exists():
                current_path.unlink()
        elif current_path != input_path:
            # Only silence was trimmed, rename to final output
            current_path.rename(fixed_path)

        report.fixed_path = fixed_path

    return report
