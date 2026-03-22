"""Tests for the full pipeline."""

from src.pipeline import process_track
from src.models.report import TrackReport, ActionType, Severity
from src.analyzers.loudness import analyze_loudness


class TestProcessTrack:

    def test_returns_track_report(self, loud_track):
        report = process_track(loud_track)
        assert isinstance(report, TrackReport)

    def test_loud_track_gets_fixed(self, loud_track, tmp_audio_dir):
        report = process_track(loud_track, output_dir=tmp_audio_dir)
        assert report.needs_fix
        assert report.fixed_path is not None
        assert report.fixed_path.exists()

    def test_fixed_file_closer_to_target(self, loud_track, tmp_audio_dir):
        report = process_track(loud_track, output_dir=tmp_audio_dir)
        fixed_loudness = analyze_loudness(report.fixed_path)
        # Fixed file should be closer to -14 LUFS than the original
        original_distance = abs(report.loudness.integrated_lufs - (-14.0))
        fixed_distance = abs(fixed_loudness.integrated_lufs - (-14.0))
        assert fixed_distance < original_distance

    def test_normal_track_not_fixed(self, normal_track):
        report = process_track(normal_track)
        assert not report.needs_fix
        assert report.fixed_path is None

    def test_platform_predictions_populated(self, loud_track):
        report = process_track(loud_track)
        assert len(report.platform_predictions) > 0
