"""Decision engine — determines what to fix vs. flag vs. ignore.

Pure functions. No side effects. Takes analysis data in, returns decisions out.
This separation exists so the decision logic can grow independently of the
analysis and remediation layers.
"""

from src.models.report import (
    LoudnessReport,
    SilenceReport,
    ClippingReport,
    PlatformPrediction,
    Severity,
    ActionType,
)
from src.platform_specs import PlatformSpec, PLATFORMS


def classify_severity(
    loudness_delta_db: float,
    true_peak_compliant: bool,
) -> Severity:
    """Determine severity from a single platform's perspective.

    Args:
        loudness_delta_db: How much the platform will adjust. Negative = turned down.
        true_peak_compliant: Whether true peak is within the platform's limit.

    Returns:
        Severity level for this platform.
    """
    if not true_peak_compliant:
        return Severity.FAIL

    abs_delta = abs(loudness_delta_db)
    if abs_delta > 3.0:
        return Severity.FAIL
    if abs_delta > 1.0:
        return Severity.WARNING
    return Severity.PASS


def build_platform_predictions(
    loudness: LoudnessReport,
) -> list[PlatformPrediction]:
    """Generate predictions for every platform from a loudness report.

    Args:
        loudness: Measured loudness data for a track.

    Returns:
        List of per-platform predictions sorted by severity (worst first).
    """
    predictions = []

    for spec in PLATFORMS.values():
        delta = spec.loudness_delta(loudness.integrated_lufs)
        headroom = spec.max_true_peak_dbtp - loudness.true_peak_dbtp
        compliant = loudness.true_peak_dbtp <= spec.max_true_peak_dbtp

        severity = classify_severity(delta, compliant)

        predictions.append(
            PlatformPrediction(
                platform_name=spec.name,
                target_lufs=spec.target_lufs,
                loudness_delta_db=round(delta, 1),
                true_peak_compliant=compliant,
                true_peak_headroom_db=round(headroom, 1),
                severity=severity,
            )
        )

    severity_order = {Severity.FAIL: 0, Severity.WARNING: 1, Severity.PASS: 2}
    predictions.sort(key=lambda p: severity_order[p.severity])

    return predictions


def decide_actions(
    loudness: LoudnessReport,
    predictions: list[PlatformPrediction],
    target_lufs: float = -14.0,
    silence: SilenceReport | None = None,
    clipping: ClippingReport | None = None,
) -> list[ActionType]:
    """Decide what actions to take based on analysis results.

    Args:
        loudness: Measured loudness data.
        predictions: Per-platform predictions.
        target_lufs: LUFS target for auto-fix (default: Spotify's -14).
        silence: Silence analysis data, if available.
        clipping: Clipping analysis data, if available.

    Returns:
        List of actions the system should take.
    """
    actions = []

    lufs_distance = abs(loudness.integrated_lufs - target_lufs)
    has_loudness_issue = lufs_distance > 1.0
    has_peak_issue = any(not p.true_peak_compliant for p in predictions)
    has_silence_issue = silence is not None and silence.needs_trim
    has_clipping = clipping is not None and clipping.has_clipping

    if has_loudness_issue or has_peak_issue or has_silence_issue:
        actions.append(ActionType.AUTO_FIX)

    # Clipping is flagged for review — auto-fix could destroy intentional distortion
    if has_clipping:
        actions.append(ActionType.FLAG_FOR_REVIEW)

    if not actions:
        actions.append(ActionType.NO_ACTION)

    return actions
