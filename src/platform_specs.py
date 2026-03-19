"""Streaming platform loudness targets and encoding specs.

Single source of truth for what each platform expects.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PlatformSpec:
    """Loudness and encoding specs for a streaming platform."""

    name: str
    target_lufs: float
    max_true_peak_dbtp: float
    normalizes_up: bool  # whether platform boosts quiet tracks
    codec: str
    bitrate: str

    def loudness_delta(self, measured_lufs: float) -> float:
        """How much the platform will adjust this track.

        Negative = turned down. Positive = turned up.
        Returns 0.0 if platform doesn't normalize up and track is quieter.
        """
        delta = self.target_lufs - measured_lufs
        if delta > 0 and not self.normalizes_up:
            return 0.0
        return delta


PLATFORMS: dict[str, PlatformSpec] = {
    "spotify": PlatformSpec(
        name="Spotify",
        target_lufs=-14.0,
        max_true_peak_dbtp=-1.0,
        normalizes_up=True,
        codec="Ogg Vorbis",
        bitrate="160-320 kbps",
    ),
    "apple_music": PlatformSpec(
        name="Apple Music",
        target_lufs=-16.0,
        max_true_peak_dbtp=-1.0,
        normalizes_up=False,
        codec="AAC",
        bitrate="256 kbps",
    ),
    "youtube": PlatformSpec(
        name="YouTube",
        target_lufs=-14.0,
        max_true_peak_dbtp=-1.0,
        normalizes_up=False,
        codec="Opus / AAC",
        bitrate="128-256 kbps",
    ),
    "tidal": PlatformSpec(
        name="Tidal",
        target_lufs=-14.0,
        max_true_peak_dbtp=-1.0,
        normalizes_up=False,
        codec="FLAC / AAC",
        bitrate="Lossless / 320 kbps",
    ),
    "amazon_music": PlatformSpec(
        name="Amazon Music",
        target_lufs=-14.0,
        max_true_peak_dbtp=-2.0,
        codec="FLAC / AAC",
        bitrate="Lossless / 256 kbps",
        normalizes_up=False,
    ),
}
