"""Pipeline — orchestrates analyze → decide → fix.

Single entry point for processing a track end-to-end.
"""

import soundfile as sf
from pathlib import Path

from src.models.report import TrackReport
from src.analyzers.loudness import analyze_loudness
from src.engine.rules import build_platform_predictions, decide_actions
from src.remediation.loudness import fix_loudness
from src.models.report import ActionType


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

    # Predict
    predictions = build_platform_predictions(loudness)

    # Decide
    actions = decide_actions(loudness, predictions, target_lufs)

    # Build report
    report = TrackReport(
        source_path=input_path,
        sample_rate=info.samplerate,
        channels=info.channels,
        duration_seconds=info.duration,
        loudness=loudness,
        platform_predictions=predictions,
        actions=actions,
    )

    # Fix if needed
    if report.needs_fix:
        stem = input_path.stem
        fixed_name = f"{stem}_fixed.wav"
        fixed_path = output_dir / fixed_name

        fix_loudness(
            input_path=input_path,
            output_path=fixed_path,
            target_lufs=target_lufs,
        )
        report.fixed_path = fixed_path

    return report
