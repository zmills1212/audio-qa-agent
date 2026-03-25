"""Pipeline — orchestrates analyze → decide → fix.

Single entry point for processing a track end-to-end.
"""

import soundfile as sf
from pathlib import Path

from src.models.report import TrackReport, ActionType
from src.analyzers.loudness import analyze_loudness
from src.analyzers.silence import analyze_silence
from src.analyzers.clipping import analyze_clipping
from src.engine.rules import build_platform_predictions, decide_actions
from src.remediation.loudness import fix_loudness
from src.remediation.silence import trim_silence
from src.platform_specs import PLATFORMS
from src.utils.audio_io import prepare_audio


def strictest_peak_ceiling() -> float:
    """Return the most restrictive true peak limit across all platforms."""
    return min(spec.max_true_peak_dbtp for spec in PLATFORMS.values())


def process_track(
    input_path: str | Path,
    output_dir: str | Path | None = None,
    target_lufs: float = -14.0,
) -> TrackReport:
    """Analyze a track, decide what to fix, and apply corrections.

    Accepts any supported audio format — MP4, M4A, MOV, MP3, WAV, FLAC, etc.
    Non-native formats are converted to WAV via ffmpeg automatically.

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

    # Convert if needed (MP4, M4A, etc. → WAV)
    audio_path, was_converted = prepare_audio(input_path, work_dir=output_dir)

    # Load file info
    info = sf.info(str(audio_path))

    # Analyze
    loudness = analyze_loudness(audio_path)
    silence = analyze_silence(audio_path)
    clipping = analyze_clipping(audio_path)

    # Predict
    predictions = build_platform_predictions(loudness)

    # Decide
    actions = decide_actions(loudness, predictions, target_lufs, silence, clipping)

    # Build report — source_path stays as the original input for display
    report = TrackReport(
        source_path=input_path,
        sample_rate=info.samplerate,
        channels=info.channels,
        duration_seconds=info.duration,
        loudness=loudness,
        silence=silence,
        clipping=clipping,
        platform_predictions=predictions,
        actions=actions,
    )

    # Fix if needed
    if report.needs_fix:
        stem = input_path.stem
        fixed_name = f"{stem}_fixed.wav"
        fixed_path = output_dir / fixed_name

        current_path = audio_path

        # Trim silence first (if needed)
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
            if current_path != audio_path and current_path.exists():
                current_path.unlink()
        elif current_path != audio_path:
            current_path.rename(fixed_path)

        # Clean up intermediate converted file if we created one
        if was_converted and audio_path != fixed_path and audio_path.exists():
            audio_path.unlink()

        report.fixed_path = fixed_path

    return report
