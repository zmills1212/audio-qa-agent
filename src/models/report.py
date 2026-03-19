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
class PlatformPrediction:
    """What a specific platform will do to this track."""

    platform_name: str
    target_lufs: float
    loudness_delta_db: float  # negative = turned down
    true_peak_compliant: bool
    true_peak_headroom_db: float  # how far below the limit
    severity: Severity


@dataclass
class TrackReport:
    """Complete analysis report for a single track."""

    source_path: Path
    sample_rate: int
    channels: int
    duration_seconds: float
    loudness: LoudnessReport | None = None
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
