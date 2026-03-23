"""Tests for clipping analyzer."""

import numpy as np
import soundfile as sf
import pytest
from pathlib import Path

from src.analyzers.clipping import analyze_clipping, find_clip_regions
from src.models.report import ClippingReport, Severity


def write_clipped_wav(path: Path, clip_length: int = 10) -> Path:
    """Create a track with intentional hard clipping."""
    sr = 44100
    t = np.linspace(0, 2.0, sr * 2, endpoint=False, dtype=np.float32)
    samples = 1.5 * np.sin(2 * np.pi * 440 * t)  # overdrive
    samples = np.clip(samples, -1.0, 1.0)
    sf.write(str(path), samples[:, np.newaxis], sr, subtype="PCM_24")
    return path


def write_clean_wav(path: Path) -> Path:
    """Create a clean track with no clipping."""
    sr = 44100
    t = np.linspace(0, 2.0, sr * 2, endpoint=False, dtype=np.float32)
    samples = 0.5 * np.sin(2 * np.pi * 440 * t)
    sf.write(str(path), samples[:, np.newaxis], sr, subtype="PCM_24")
    return path


class TestAnalyzeClipping:

    def test_returns_clipping_report(self, tmp_path):
        path = write_clipped_wav(tmp_path / "clipped.wav")
        report = analyze_clipping(path)
        assert isinstance(report, ClippingReport)

    def test_detects_clipping(self, tmp_path):
        path = write_clipped_wav(tmp_path / "clipped.wav")
        report = analyze_clipping(path)
        assert report.has_clipping
        assert report.clip_count > 0
        assert report.clipped_samples > 0

    def test_clean_track_no_clipping(self, tmp_path):
        path = write_clean_wav(tmp_path / "clean.wav")
        report = analyze_clipping(path)
        assert not report.has_clipping
        assert report.clip_count == 0
        assert report.severity == Severity.PASS

    def test_clipped_track_severity_fail(self, tmp_path):
        path = write_clipped_wav(tmp_path / "clipped.wav")
        report = analyze_clipping(path)
        assert report.severity == Severity.FAIL

    def test_clip_percentage_nonzero(self, tmp_path):
        path = write_clipped_wav(tmp_path / "clipped.wav")
        report = analyze_clipping(path)
        assert report.clip_percentage > 0.0

    def test_stereo_clipping(self, tmp_path):
        sr = 44100
        t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
        left = np.clip(1.5 * np.sin(2 * np.pi * 440 * t), -1.0, 1.0)
        right = 0.5 * np.sin(2 * np.pi * 440 * t)  # clean
        samples = np.column_stack([left, right])
        path = tmp_path / "stereo_clip.wav"
        sf.write(str(path), samples, sr, subtype="PCM_24")
        report = analyze_clipping(path)
        assert report.has_clipping
        # All clip regions should be on channel 0
        assert all(r.channel == 0 for r in report.clip_regions)

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            analyze_clipping("/nonexistent/track.wav")


class TestFindClipRegions:

    def test_no_clipping_returns_empty(self):
        samples = np.array([0.5, 0.6, 0.7, 0.5, 0.3], dtype=np.float32)
        regions = find_clip_regions(samples, channel=0)
        assert len(regions) == 0

    def test_short_clip_ignored(self):
        """Two consecutive clipped samples should not count (min is 3)."""
        samples = np.array([0.5, 1.0, 1.0, 0.5], dtype=np.float32)
        regions = find_clip_regions(samples, channel=0, min_consecutive=3)
        assert len(regions) == 0

    def test_three_consecutive_detected(self):
        samples = np.array([0.5, 1.0, 1.0, 1.0, 0.5], dtype=np.float32)
        regions = find_clip_regions(samples, channel=0, min_consecutive=3)
        assert len(regions) == 1
        assert regions[0].length == 3

    def test_multiple_regions(self):
        samples = np.array(
            [0.5, 1.0, 1.0, 1.0, 0.5, 0.3, -1.0, -1.0, -1.0, -1.0, 0.2],
            dtype=np.float32,
        )
        regions = find_clip_regions(samples, channel=0, min_consecutive=3)
        assert len(regions) == 2

    def test_negative_clipping_detected(self):
        samples = np.array([0.5, -1.0, -1.0, -1.0, 0.5], dtype=np.float32)
        regions = find_clip_regions(samples, channel=0, min_consecutive=3)
        assert len(regions) == 1
