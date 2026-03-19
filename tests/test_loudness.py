"""Tests for loudness analyzer."""

import math
from src.analyzers.loudness import (
    analyze_loudness,
    measure_true_peak,
    measure_loudness_range,
)
from src.models.report import LoudnessReport


class TestAnalyzeLoudness:
    """Integration tests for the full loudness analysis."""

    def test_returns_loudness_report(self, normal_track):
        report = analyze_loudness(normal_track)
        assert isinstance(report, LoudnessReport)

    def test_loud_track_above_spotify_target(self, loud_track):
        report = analyze_loudness(loud_track)
        assert report.integrated_lufs > -14.0, "Loud track should be above Spotify's -14 LUFS"

    def test_quiet_track_below_spotify_target(self, quiet_track):
        report = analyze_loudness(quiet_track)
        assert report.integrated_lufs < -14.0, "Quiet track should be below Spotify's -14 LUFS"

    def test_true_peak_below_sample_peak_is_possible(self, normal_track):
        """True peak can exceed sample peak due to inter-sample peaks."""
        report = analyze_loudness(normal_track)
        # True peak should be >= sample peak (oversampling finds hidden peaks)
        assert report.true_peak_dbtp >= report.sample_peak_dbfs - 0.5

    def test_clipped_track_has_high_true_peak(self, clipped_track):
        report = analyze_loudness(clipped_track)
        assert report.true_peak_dbtp >= -0.5, "Clipped track should have true peak near 0 dBTP"

    def test_stereo_handled(self, stereo_track):
        report = analyze_loudness(stereo_track)
        assert math.isfinite(report.integrated_lufs)
        assert math.isfinite(report.true_peak_dbtp)

    def test_loudness_range_nonnegative(self, normal_track):
        report = analyze_loudness(normal_track)
        assert report.loudness_range_lu >= 0.0

    def test_file_not_found_raises(self):
        import pytest
        with pytest.raises(FileNotFoundError):
            analyze_loudness("/nonexistent/track.wav")


class TestMeasureTruePeak:
    """Unit tests for true peak measurement."""

    def test_silent_audio_returns_negative_inf(self):
        import numpy as np
        silence = np.zeros((44100, 1), dtype=np.float32)
        result = measure_true_peak(silence, 44100)
        assert result == float("-inf")

    def test_full_scale_sine_near_zero_dbtp(self):
        import numpy as np
        t = np.linspace(0, 1.0, 44100, endpoint=False, dtype=np.float32)
        sine = np.sin(2 * np.pi * 997 * t)[:, np.newaxis]
        result = measure_true_peak(sine, 44100)
        assert result > -1.0, "Full-scale sine should have true peak near 0 dBTP"
        assert result < 1.0, "True peak shouldn't significantly exceed 0 dBTP"
