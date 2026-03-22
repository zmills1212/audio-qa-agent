"""Tests for the decision engine."""

from src.engine.rules import classify_severity, build_platform_predictions, decide_actions
from src.models.report import LoudnessReport, Severity, ActionType


def make_loudness(lufs=-10.0, true_peak=-0.5):
    """Helper to create a LoudnessReport with sensible defaults."""
    return LoudnessReport(
        integrated_lufs=lufs,
        true_peak_dbtp=true_peak,
        loudness_range_lu=6.0,
        short_term_max_lufs=lufs + 3.0,
        sample_peak_dbfs=true_peak + 0.5,
    )


class TestClassifySeverity:

    def test_compliant_and_close_is_pass(self):
        assert classify_severity(loudness_delta_db=-0.5, true_peak_compliant=True) == Severity.PASS

    def test_moderate_delta_is_warning(self):
        assert classify_severity(loudness_delta_db=-2.0, true_peak_compliant=True) == Severity.WARNING

    def test_large_delta_is_fail(self):
        assert classify_severity(loudness_delta_db=-5.0, true_peak_compliant=True) == Severity.FAIL

    def test_peak_noncompliant_is_always_fail(self):
        assert classify_severity(loudness_delta_db=0.0, true_peak_compliant=False) == Severity.FAIL


class TestBuildPlatformPredictions:

    def test_returns_predictions_for_all_platforms(self):
        loudness = make_loudness(lufs=-10.0)
        predictions = build_platform_predictions(loudness)
        names = {p.platform_name for p in predictions}
        assert "Spotify" in names
        assert "Apple Music" in names
        assert "YouTube" in names

    def test_loud_track_gets_negative_delta(self):
        loudness = make_loudness(lufs=-8.0)
        predictions = build_platform_predictions(loudness)
        spotify = next(p for p in predictions if p.platform_name == "Spotify")
        assert spotify.loudness_delta_db < 0

    def test_quiet_track_not_boosted_by_apple(self):
        loudness = make_loudness(lufs=-20.0)
        predictions = build_platform_predictions(loudness)
        apple = next(p for p in predictions if p.platform_name == "Apple Music")
        assert apple.loudness_delta_db == 0.0  # Apple doesn't normalize up


class TestDecideActions:

    def test_compliant_track_gets_no_action(self):
        loudness = make_loudness(lufs=-14.0, true_peak=-2.0)
        predictions = build_platform_predictions(loudness)
        actions = decide_actions(loudness, predictions)
        assert ActionType.NO_ACTION in actions

    def test_loud_track_gets_auto_fix(self):
        loudness = make_loudness(lufs=-8.0, true_peak=-0.5)
        predictions = build_platform_predictions(loudness)
        actions = decide_actions(loudness, predictions)
        assert ActionType.AUTO_FIX in actions

    def test_hot_peak_gets_auto_fix(self):
        loudness = make_loudness(lufs=-14.0, true_peak=0.5)
        predictions = build_platform_predictions(loudness)
        actions = decide_actions(loudness, predictions)
        assert ActionType.AUTO_FIX in actions
