"""Tests for silence analyzer and remediation."""

import numpy as np
import soundfile as sf
import pytest
from pathlib import Path

from src.analyzers.silence import analyze_silence, compute_rms_frames
from src.remediation.silence import trim_silence
from src.models.report import SilenceReport


def write_silence_track(path: Path, leading_s: float, trailing_s: float) -> Path:
    """Create a track with specific leading/trailing silence."""
    sr = 44100
    leading = np.zeros(int(sr * leading_s), dtype=np.float32)
    content = 0.5 * np.sin(
        2 * np.pi * 440 * np.linspace(0, 2.0, int(sr * 2.0), endpoint=False, dtype=np.float32)
    )
    trailing = np.zeros(int(sr * trailing_s), dtype=np.float32)
    samples = np.concatenate([leading, content, trailing])[:, np.newaxis]
    sf.write(str(path), samples, sr, subtype="PCM_24")
    return path


class TestAnalyzeSilence:

    def test_returns_silence_report(self, tmp_path):
        path = write_silence_track(tmp_path / "test.wav", 0.5, 0.5)
        report = analyze_silence(path)
        assert isinstance(report, SilenceReport)

    def test_detects_leading_silence(self, tmp_path):
        path = write_silence_track(tmp_path / "test.wav", 1.0, 0.0)
        report = analyze_silence(path)
        assert report.leading_silence_ms > 900  # ~1000ms, allow margin
        assert report.leading_trimmed

    def test_detects_trailing_silence(self, tmp_path):
        path = write_silence_track(tmp_path / "test.wav", 0.0, 1.5)
        report = analyze_silence(path)
        assert report.trailing_silence_ms > 1400
        assert report.trailing_trimmed

    def test_no_silence_detected_on_clean_track(self, normal_track):
        report = analyze_silence(normal_track)
        assert not report.needs_trim

    def test_total_silence(self, tmp_path):
        path = write_silence_track(tmp_path / "test.wav", 0.5, 0.8)
        report = analyze_silence(path)
        assert report.total_silence_ms > 1200

    def test_short_silence_ignored(self, tmp_path):
        path = write_silence_track(tmp_path / "test.wav", 0.05, 0.05)
        report = analyze_silence(path, min_silence_ms=100.0)
        assert not report.needs_trim

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            analyze_silence("/nonexistent/track.wav")


class TestTrimSilence:

    def test_trimmed_file_shorter(self, tmp_path):
        original = write_silence_track(tmp_path / "original.wav", 1.0, 1.0)
        trimmed_path = tmp_path / "trimmed.wav"
        trim_silence(original, trimmed_path)
        orig_samples, _ = sf.read(str(original))
        trim_samples, _ = sf.read(str(trimmed_path))
        assert len(trim_samples) < len(orig_samples)

    def test_trimmed_file_exists(self, tmp_path):
        original = write_silence_track(tmp_path / "original.wav", 0.5, 0.5)
        trimmed_path = tmp_path / "trimmed.wav"
        result = trim_silence(original, trimmed_path)
        assert result.exists()

    def test_no_trim_when_silence_below_threshold(self, tmp_path):
        original = write_silence_track(tmp_path / "original.wav", 0.05, 0.05)
        trimmed_path = tmp_path / "trimmed.wav"
        trim_silence(original, trimmed_path, min_silence_ms=100.0)
        orig_samples, _ = sf.read(str(original))
        trim_samples, _ = sf.read(str(trimmed_path))
        assert len(trim_samples) == len(orig_samples)


class TestComputeRmsFrames:

    def test_silent_audio_has_zero_rms(self):
        silence = np.zeros(44100, dtype=np.float32)
        rms = compute_rms_frames(silence, 44100)
        assert np.all(rms == 0.0)

    def test_loud_audio_has_nonzero_rms(self):
        t = np.linspace(0, 1.0, 44100, endpoint=False, dtype=np.float32)
        sine = np.sin(2 * np.pi * 440 * t)
        rms = compute_rms_frames(sine, 44100)
        assert np.all(rms > 0.0)
