"""CLI entry point — the demo interface.

Usage:
    python -m src.cli path/to/track.wav
    python -m src.cli path/to/track.wav --target-lufs -16
    python -m src.cli path/to/track.wav --output-dir ./fixed
"""

import argparse
import sys
from pathlib import Path

from src.pipeline import process_track
from src.models.report import Severity, ActionType


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

    # Header
    lines.append("")
    lines.append(f"  Track:      {report.source_path.name}")
    lines.append(f"  Duration:   {report.duration_seconds:.1f}s")
    lines.append(f"  Sample Rate: {report.sample_rate} Hz | Channels: {report.channels}")
    lines.append("")

    # Loudness measurements
    lines.append("  ── Loudness Measurements ──")
    lines.append(f"  Integrated LUFS:   {loud.integrated_lufs:.1f}")
    lines.append(f"  True Peak:         {loud.true_peak_dbtp:.1f} dBTP")
    lines.append(f"  Sample Peak:       {loud.sample_peak_dbfs:.1f} dBFS")
    lines.append(f"  Loudness Range:    {loud.loudness_range_lu:.1f} LU")
    lines.append(f"  Short-term Max:    {loud.short_term_max_lufs:.1f} LUFS")
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

    # Action taken
    if ActionType.AUTO_FIX in report.actions:
        lines.append(f"  🔧 Fixed version saved: {report.fixed_path.name}")
    else:
        lines.append("  ✅ No issues found — track is platform-ready.")

    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Audio QA Agent — analyze and fix tracks for streaming platforms",
    )
    parser.add_argument("track", type=Path, help="Path to audio file")
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
