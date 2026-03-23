"""Data models for analysis reports and platform predictions.

These are the contracts between every module in the system.
Analyzers produce them, the engine consumes them, remediation acts on them.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class Severity(Enum):
    """How urgent is this issue."""

    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"


class ActionType(Enum):
    """What the engine decided to do."""

    NO_ACTION = "no_action"
    AUTO_FIX = "auto_fix"
    FLAG_FOR_REVIEW = "flag_for_review"


@dataclass(frozen=True)
class LoudnessReport:
    """Output of the loudness analyzer."""

    integrated_lufs: float
    true_peak_dbtp: float
    loudness_range_lu: float
    short_term_max_lufs: float
    sample_peak_dbfs: float


@dataclass(frozen=True)
class SilenceReport:
    """Output of the silence analyzer."""

    leading_silence_ms: float
    trailing_silence_ms: float
    leading_trimmed: bool
    trailing_trimmed: bool

    @property
    def total_silence_ms(self) -> float:
        return self.leading_silence_ms + self.trailing_silence_ms

    @property
    def needs_trim(self) -> bool:
        return self.leading_trimmed or self.trailing_trimmed


@dataclass(frozen=True)
class ClipRegion:
    """A contiguous region of clipped samples."""

    start_sample: int
    end_sample: int
    channel: int

    @property
    def length(self) -> int:
        return self.end_sample - self.start_sample


@dataclass(frozen=True)
class ClippingReport:
    """Output of the clipping analyzer."""

    clip_count: int  # total number of clipped regions
    clipped_samples: int  # total individual samples at the rail
    total_samples: int
    clip_regions: list[ClipRegion]
    peak_dbfs: float  # sample peak in dBFS
    severity: Severity

    @property
    def clip_percentage(self) -> float:
        if self.total_samples == 0:
            return 0.0
        return (self.clipped_samples / self.total_samples) * 100

    @property
    def has_clipping(self) -> bool:
        return self.clip_count > 0


@dataclass(frozen=True)
class PlatformPrediction:
    """What a specific platform will do to this track."""

    platform_name: str
    target_lufs: float
    loudness_delta_db: float
    true_peak_compliant: bool
    true_peak_headroom_db: float
    severity: Severity


@dataclass
class TrackReport:
    """Complete analysis report for a single track."""

    source_path: Path
    sample_rate: int
    channels: int
    duration_seconds: float
    loudness: LoudnessReport | None = None
    silence: SilenceReport | None = None
    clipping: ClippingReport | None = None
    platform_predictions: list[PlatformPrediction] = field(default_factory=list)
    actions: list[ActionType] = field(default_factory=list)
    fixed_path: Path | None = None

    @property
    def needs_fix(self) -> bool:
        return ActionType.AUTO_FIX in self.actions

    @property
    def worst_severity(self) -> Severity:
        if not self.platform_predictions:
            return Severity.PASS
        severities = [p.severity for p in self.platform_predictions]
        if Severity.FAIL in severities:
            return Severity.FAIL
        if Severity.WARNING in severities:
            return Severity.WARNING
        return Severity.PASS
