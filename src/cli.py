"""CLI entry point — the demo interface.

Usage:
    python -m src path/to/track.wav
    python -m src path/to/track.mp4
    python -m src path/to/track.m4a --target-lufs -16
    python -m src path/to/track.wav --output-dir ./fixed
"""

import argparse
import sys
from pathlib import Path

from src.pipeline import process_track
from src.models.report import Severity, ActionType
from src.utils.audio_io import ALL_SUPPORTED


def severity_icon(severity: Severity) -> str:
    return {
        Severity.PASS: "✅",
        Severity.WARNING: "⚠️",
        Severity.FAIL: "❌",
    }[severity]


def format_report(report) -> str:
    """Format a TrackReport as a readable CLI output."""
    lines = []
    loud = report.loudness
    silence = report.silence
    clipping = report.clipping

    # Header
    lines.append("")
    lines.append(f"  Track:       {report.source_path.name}")
    lines.append(f"  Duration:    {report.duration_seconds:.1f}s")
    lines.append(f"  Sample Rate: {report.sample_rate} Hz | Channels: {report.channels}")
    lines.append("")

    # Loudness
    lines.append("  ── Loudness ──")
    lines.append(f"  Integrated LUFS:   {loud.integrated_lufs:.1f}")
    lines.append(f"  True Peak:         {loud.true_peak_dbtp:.1f} dBTP")
    lines.append(f"  Sample Peak:       {loud.sample_peak_dbfs:.1f} dBFS")
    lines.append(f"  Loudness Range:    {loud.loudness_range_lu:.1f} LU")
    lines.append(f"  Short-term Max:    {loud.short_term_max_lufs:.1f} LUFS")
    lines.append("")

    # Silence
    if silence:
        lines.append("  ── Silence ──")
        leading_s = silence.leading_silence_ms / 1000
        trailing_s = silence.trailing_silence_ms / 1000
        lead_flag = " ← trim" if silence.leading_trimmed else ""
        trail_flag = " ← trim" if silence.trailing_trimmed else ""
        lines.append(f"  Leading:   {leading_s:.2f}s{lead_flag}")
        lines.append(f"  Trailing:  {trailing_s:.2f}s{trail_flag}")
        lines.append("")

    # Clipping
    if clipping:
        lines.append("  ── Clipping ──")
        icon = severity_icon(clipping.severity)
        if clipping.has_clipping:
            lines.append(f"  {icon} {clipping.clip_count} clipped regions detected")
            lines.append(f"     {clipping.clipped_samples} samples affected ({clipping.clip_percentage:.4f}%)")
            lines.append(f"     Sample peak: {clipping.peak_dbfs:.1f} dBFS")
        else:
            lines.append(f"  {icon} No clipping detected")
        lines.append("")

    # Platform predictions
    lines.append("  ── Platform Predictions ──")

    for p in report.platform_predictions:
        icon = severity_icon(p.severity)

        if p.loudness_delta_db < 0:
            delta_str = f"reduced by {abs(p.loudness_delta_db):.1f} dB"
        elif p.loudness_delta_db > 0:
            delta_str = f"boosted by {p.loudness_delta_db:.1f} dB"
        else:
            delta_str = "no change"

        peak_str = "OK" if p.true_peak_compliant else "CLIP RISK"

        lines.append(f"  {icon} {p.platform_name:<14} Loudness: {delta_str:<22} Peak: {peak_str}")

    lines.append("")

    # Actions
    if ActionType.AUTO_FIX in report.actions:
        fixes = []
        if silence and silence.needs_trim:
            fixes.append("trimmed silence")
        if any(not p.true_peak_compliant for p in report.platform_predictions):
            fixes.append("limited peaks")
        lufs_distance = abs(loud.integrated_lufs - (-14.0))
        if lufs_distance > 1.0:
            fixes.append("normalized loudness")
        fix_desc = ", ".join(fixes) if fixes else "applied corrections"
        lines.append(f"  🔧 Fixed ({fix_desc}): {report.fixed_path.name}")

    if ActionType.FLAG_FOR_REVIEW in report.actions:
        lines.append("  ⚠️  Clipping detected — review recommended (may be intentional distortion)")

    if report.actions == [ActionType.NO_ACTION]:
        lines.append("  ✅ No issues found — track is platform-ready.")

    lines.append("")
    return "\n".join(lines)


def main():
    extensions = ", ".join(sorted(ALL_SUPPORTED))
    parser = argparse.ArgumentParser(
        description="Audio QA Agent — analyze and fix tracks for streaming platforms",
    )
    parser.add_argument("track", type=Path, help=f"Path to audio file ({extensions})")
    parser.add_argument(
        "--target-lufs",
        type=float,
        default=-14.0,
        help="Target LUFS for normalization (default: -14.0)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for fixed files (default: same as input)",
    )
    args = parser.parse_args()

    if not args.track.exists():
        print(f"\n  Error: File not found — {args.track}\n", file=sys.stderr)
        sys.exit(1)

    print(f"\n  Analyzing {args.track.name}...")
    report = process_track(args.track, args.output_dir, args.target_lufs)
    print(format_report(report))


if __name__ == "__main__":
    main()
