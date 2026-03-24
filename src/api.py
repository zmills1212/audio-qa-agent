"""FastAPI backend — serves the web UI and handles track processing.

Run with: uvicorn src.api:app --reload
"""

import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.pipeline import process_track
from src.models.report import Severity, ActionType

app = FastAPI(title="Audio QA Agent", version="0.1.0")

# Temp directory for uploaded and processed files
WORK_DIR = Path(tempfile.mkdtemp(prefix="audio_qa_"))


def severity_str(s: Severity) -> str:
    return {"pass": "pass", "warning": "warning", "fail": "fail"}[s.value]


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the single-page frontend."""
    html_path = Path(__file__).parent / "static" / "index.html"
    return HTMLResponse(html_path.read_text())


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...), target_lufs: float = -14.0):
    """Upload a track, analyze it, and return the report + fixed file info."""

    if not file.filename:
        raise HTTPException(400, "No file provided")

    # Save upload to temp directory
    upload_path = WORK_DIR / file.filename
    with open(upload_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        report = process_track(upload_path, output_dir=WORK_DIR, target_lufs=target_lufs)
    except Exception as e:
        raise HTTPException(422, f"Analysis failed: {e}")

    loud = report.loudness
    silence = report.silence
    clipping = report.clipping

    # Build response
    response = {
        "track": {
            "name": report.source_path.name,
            "duration": round(report.duration_seconds, 1),
            "sample_rate": report.sample_rate,
            "channels": report.channels,
        },
        "loudness": {
            "integrated_lufs": round(loud.integrated_lufs, 1),
            "true_peak_dbtp": round(loud.true_peak_dbtp, 1),
            "sample_peak_dbfs": round(loud.sample_peak_dbfs, 1),
            "loudness_range_lu": round(loud.loudness_range_lu, 1),
            "short_term_max_lufs": round(loud.short_term_max_lufs, 1),
        },
        "silence": {
            "leading_ms": silence.leading_silence_ms,
            "trailing_ms": silence.trailing_silence_ms,
            "leading_trimmed": silence.leading_trimmed,
            "trailing_trimmed": silence.trailing_trimmed,
            "needs_trim": silence.needs_trim,
        } if silence else None,
        "clipping": {
            "has_clipping": clipping.has_clipping,
            "clip_count": clipping.clip_count,
            "clipped_samples": clipping.clipped_samples,
            "clip_percentage": round(clipping.clip_percentage, 4),
            "peak_dbfs": round(clipping.peak_dbfs, 1),
            "severity": severity_str(clipping.severity),
        } if clipping else None,
        "platforms": [
            {
                "name": p.platform_name,
                "target_lufs": p.target_lufs,
                "delta_db": p.loudness_delta_db,
                "peak_compliant": p.true_peak_compliant,
                "headroom_db": p.true_peak_headroom_db,
                "severity": severity_str(p.severity),
            }
            for p in report.platform_predictions
        ],
        "actions": [a.value for a in report.actions],
        "fixed_filename": report.fixed_path.name if report.fixed_path else None,
    }

    return JSONResponse(response)


@app.get("/api/download/{filename}")
async def download(filename: str):
    """Download a processed file."""
    file_path = WORK_DIR / filename
    if not file_path.exists():
        raise HTTPException(404, "File not found")
    return FileResponse(file_path, filename=filename, media_type="audio/wav")
