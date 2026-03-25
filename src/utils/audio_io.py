"""Audio I/O utilities — format detection and conversion.

Handles the gap between what users upload (MP4, M4A, MOV, webm, etc.)
and what the analysis pipeline needs (PCM audio readable by soundfile).

If a file is natively supported, it passes through untouched.
If not, ffmpeg converts it to 24-bit WAV transparently.
"""

import subprocess
import shutil
from pathlib import Path

import soundfile as sf


# Formats soundfile can read directly (libsndfile-backed)
NATIVE_EXTENSIONS = {".wav", ".flac", ".ogg", ".aif", ".aiff"}

# Formats that need ffmpeg conversion
CONVERT_EXTENSIONS = {".mp4", ".m4a", ".mov", ".webm", ".mp3", ".aac", ".wma", ".opus"}

ALL_SUPPORTED = NATIVE_EXTENSIONS | CONVERT_EXTENSIONS


def is_native(path: Path) -> bool:
    """Check if soundfile can read this format directly."""
    return path.suffix.lower() in NATIVE_EXTENSIONS


def needs_conversion(path: Path) -> bool:
    """Check if this format needs ffmpeg conversion."""
    return path.suffix.lower() in CONVERT_EXTENSIONS


def is_supported(path: Path) -> bool:
    """Check if this format is supported at all."""
    return path.suffix.lower() in ALL_SUPPORTED


def ensure_ffmpeg() -> str:
    """Return the path to ffmpeg, or raise if not found."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "ffmpeg not found. Install it with: brew install ffmpeg (macOS) "
            "or apt install ffmpeg (Linux)"
        )
    return ffmpeg


def convert_to_wav(
    input_path: Path,
    output_dir: Path | None = None,
    sample_rate: int | None = None,
) -> Path:
    """Convert any audio/video file to 24-bit WAV via ffmpeg.

    Args:
        input_path: Path to source file.
        output_dir: Where to write the WAV. Defaults to same directory as input.
        sample_rate: Force a specific sample rate. None preserves the original.

    Returns:
        Path to the converted WAV file.

    Raises:
        FileNotFoundError: If input doesn't exist.
        RuntimeError: If ffmpeg is missing or conversion fails.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    if output_dir is None:
        output_dir = input_path.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ffmpeg = ensure_ffmpeg()
    output_path = output_dir / f"{input_path.stem}.wav"

    cmd = [
        ffmpeg,
        "-y",              # overwrite without asking
        "-i", str(input_path),
        "-vn",             # strip video
        "-acodec", "pcm_s24le",
        "-f", "wav",
    ]

    if sample_rate is not None:
        cmd.extend(["-ar", str(sample_rate)])

    cmd.append(str(output_path))

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg conversion failed for {input_path.name}: {result.stderr[-500:]}"
        )

    return output_path


def prepare_audio(
    input_path: str | Path,
    work_dir: Path | None = None,
) -> tuple[Path, bool]:
    """Ensure a file is ready for the analysis pipeline.

    If the file is natively readable, returns it as-is.
    If it needs conversion, converts to WAV and returns the new path.

    Args:
        input_path: Path to the audio file.
        work_dir: Directory for converted files. Defaults to input's directory.

    Returns:
        Tuple of (path to readable audio, was_converted).

    Raises:
        ValueError: If format is not supported.
        FileNotFoundError: If file doesn't exist.
        RuntimeError: If conversion fails.
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    if not is_supported(input_path):
        raise ValueError(
            f"Unsupported format: {input_path.suffix}. "
            f"Supported: {', '.join(sorted(ALL_SUPPORTED))}"
        )

    # Try native read first — some .mp3 files work with soundfile
    try:
        sf.info(str(input_path))
        return input_path, False
    except Exception:
        pass

    # Convert via ffmpeg
    if work_dir is None:
        work_dir = input_path.parent

    wav_path = convert_to_wav(input_path, work_dir)
    return wav_path, True
