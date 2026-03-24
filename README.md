# Audio QA Agent

Autonomous audio quality assurance for music distribution.

Upload a track → see what streaming platforms will do to it → get a fixed version back.

## The Problem

Independent artists lose control of how their music sounds on streaming platforms.
Spotify, Apple Music, and YouTube each normalize loudness differently — a track
mastered at -8 LUFS gets turned down ~6 dB on Spotify, crushing dynamics the artist
intended. True peaks above -1 dBTP risk clipping after platform encoding. Most
artists discover these issues after release, or never.

## What This Does

**Analyzes** your track across three dimensions:

- **Loudness** — integrated LUFS, true peak (4x oversampled), loudness range, per ITU-R BS.1770-4
- **Clipping** — detects hard digital clipping with per-channel region tracking
- **Silence** — measures leading/trailing dead air with configurable thresholds

**Predicts** how five major platforms will handle your track:

- Spotify, Apple Music, YouTube, Tidal, Amazon Music
- Per-platform loudness adjustment and true peak compliance

**Fixes** issues automatically:

- LUFS normalization to your chosen target
- True peak limiting using the strictest ceiling across all platforms
- Silence trimming with fade to prevent clicks

## Example Output
```
$ python -m src track.wav

  Track:       track.wav
  Duration:    33.5s
  Sample Rate: 44100 Hz | Channels: 2

  ── Loudness ──
  Integrated LUFS:   -14.8
  True Peak:         -1.9 dBTP
  Sample Peak:       -1.9 dBFS
  Loudness Range:    6.6 LU
  Short-term Max:    -10.6 LUFS

  ── Silence ──
  Leading:   0.00s
  Trailing:  0.01s

  ── Clipping ──
  ✅ No clipping detected

  ── Platform Predictions ──
  ❌ Amazon Music   Loudness: no change              Peak: CLIP RISK
  ⚠️ Apple Music    Loudness: reduced by 1.2 dB      Peak: OK
  ✅ Spotify        Loudness: boosted by 0.8 dB      Peak: OK
  ✅ YouTube        Loudness: no change              Peak: OK
  ✅ Tidal          Loudness: no change              Peak: OK

  🔧 Fixed (limited peaks): track_fixed.wav
```

## Architecture
```
src/
├── cli.py                 # CLI entry point
├── pipeline.py            # Orchestration: analyze → decide → fix
├── platform_specs.py      # Streaming platform targets
├── models/
│   └── report.py          # Data contracts (TrackReport, LoudnessReport, etc.)
├── analyzers/
│   ├── loudness.py        # BS.1770-4 LUFS + true peak via 4x oversampling
│   ├── silence.py         # RMS-based silence detection
│   └── clipping.py        # Hard clip region detection
├── remediation/
│   ├── loudness.py        # LUFS normalization + peak limiting
│   └── silence.py         # Trim with fade
└── engine/
    └── rules.py           # Decision logic: fix vs. flag vs. ignore
```

The pipeline flows **analyze → decide → fix**. Analyzers produce structured reports,
the engine decides what to act on, and remediation modules apply corrections.
Clipping is flagged for human review rather than auto-fixed — it may be intentional
distortion.

## Quick Start
```bash
git clone https://github.com/zmills1212/audio-qa-agent.git
cd audio-qa-agent
python3 -m venv venv
source venv/bin/activate
pip install pyloudnorm soundfile numpy scipy

python -m src your_track.wav
python -m src your_track.wav --target-lufs -16
python -m src your_track.wav --output-dir ./fixed
```

## Running Tests
```bash
pip install pytest
pytest -v
```

49 tests covering analyzers, decision engine, remediation, and pipeline integration.
All tests use synthetic audio — no real music files required.

## Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Loudness | `pyloudnorm` | BS.1770-4 integrated LUFS measurement |
| True Peak | `scipy.signal.resample_poly` | 4x oversampling per BS.1770-4 |
| Audio I/O | `soundfile` | Read/write WAV, FLAC, MP3, etc. |
| DSP | `numpy` | Signal analysis and manipulation |
| CLI | `argparse` | Command-line interface |

## Roadmap

- [ ] Web UI — upload, visualize, compare before/after
- [ ] Codec simulation — predict artifacts from Spotify/Apple encoding
- [ ] Metadata validation — ISRC, tags, album art compliance
- [ ] Agent reasoning — LLM layer for ambiguous decisions (intentional vs. accidental distortion)
- [ ] Batch processing — analyze entire catalogs
- [ ] Per-artist learning — track patterns and suppress repeat alerts

## License

MIT
