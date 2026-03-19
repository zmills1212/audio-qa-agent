# Audio QA Agent

Autonomous audio quality assurance for music distribution.

Upload a track → see what streaming platforms will do to it → get a fixed version back.

## The Problem

Independent artists lose control of how their music sounds on streaming platforms.
Spotify, Apple Music, and YouTube each normalize loudness differently. A track mastered
at -8 LUFS gets turned down ~6 dB on Spotify — crushing dynamics the artist intended.
True peaks above -1 dBTP risk clipping after platform encoding. Most artists discover
these issues after release, or never.

## What This Does

- **Measures** integrated loudness (LUFS) and true peak (dBTP) per ITU-R BS.1770-4
- **Predicts** exactly how each platform will adjust your track
- **Fixes** loudness and peak issues automatically, preserving the original
- **Reports** a clear before/after comparison

## Quick Start
```bash
python3 -m venv venv
source venv/bin/activate
pip install -e .

python -m src.cli your_track.wav
```

## Stack

- `pyloudnorm` — BS.1770-4 loudness measurement
- `soundfile` — Audio I/O
- `scipy` — True peak detection via 4x oversampling
- `numpy` — Signal processing

## License

MIT
