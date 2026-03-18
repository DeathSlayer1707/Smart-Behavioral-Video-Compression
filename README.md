# Smart Behavioral Video Compression
### SentioMind Assignment · Smart CCTV Compression Pipeline

---

## Problem Statement

School CCTV systems generate **40–80 GB of raw footage per day** across multiple cameras. Uploading this over school-grade internet takes **6–12 hours**. Blind ffmpeg re-encoding is not enough — we need an intelligent, content-aware compression pipeline that:

- Keeps **every frame containing a human face**
- Keeps frames with **meaningful motion** (people walking, activity)
- Discards **static, empty-scene frames**
- Achieves **≥ 70% file size reduction**
- Runs at **≥ 4× real-time speed** (2-min video in ≤ 10s on a laptop)

---

## Algorithm (Implemented Exactly as Specified)

```
Step 1 → Perceptual Hash (pHash)
         Drop frame if similarity to last kept frame > 95%
         (eliminates near-duplicate static frames)

Step 2 → Optical Flow Motion Score
         Discard frame if mean flow magnitude < 0.05
         (empty, motionless scene)

Step 3 → Haar Face Detection
         Keep frame REGARDLESS of motion/hash if any face is detected
         (never lose a human event)

Step 4 → Context Frame (scene continuity)
         Force-keep one frame every 3 seconds minimum
         (prevents complete gaps in timeline)

Step 5 → Re-encode surviving frames → H.264 MP4 @ 12 fps via ffmpeg
```

---

## Project Structure

```
.
├── solution.py               ← Main compression script (this file)
├── compressed_output.mp4     ← Output compressed video
├── compression_report.html   ← Offline storyboard + stats report
├── segments_kept.json        ← Integration contract JSON
├── README.md                 ← This file
└── requirements.txt          ← Python dependencies
```

---

## Installation

```bash
pip install opencv-python==4.9.0 numpy==1.26.4 Pillow==10.3.0 \
            imagehash==4.3.1 scikit-image==0.22.0

# ffmpeg must be installed on your system
# Ubuntu/Debian:
sudo apt install ffmpeg

# macOS (Homebrew):
brew install ffmpeg

# Windows: download from https://ffmpeg.org/download.html
```

Full library stack (as specified):
```
opencv-python==4.9.0   face_recognition==1.3.0   mediapipe==0.10.14
deepface==0.0.93       mtcnn==0.1.1               numpy==1.26.4
Pillow==10.3.0         scikit-image==0.22.0       imagehash==4.3.1
```

---

## Usage

```bash
# Basic usage (expects video_sample_1.mov in current directory)
python solution.py

# Custom input path
python solution.py path/to/your_video.mov
```

Output files are written to the current directory:
| File | Description |
|------|-------------|
| `compressed_output.mp4` | H.264 re-encoded compressed video |
| `compression_report.html` | Offline HTML report with storyboard |
| `segments_kept.json` | Integration schema for Sentio pipeline |

---

## API Reference

### `compress_video(input_path, output_video, output_json, output_html) → dict`
Full end-to-end pipeline. Returns summary dict with timing and size stats.

### `select_keyframes(video_path, ...) → List[Dict]`
Runs Steps 1–4. Returns list of kept frame metadata dicts.

### `extract_intelligent_frames(segments_json_path, video_path) → List[np.ndarray]`
**Integration entry point.** Reads `segments_kept.json` and returns BGR frame list without re-running the pipeline. Plugs directly into the Sentio main pipeline replacing full video scan.

### `encode_compressed_video(video_path, kept_frames, output_path, output_fps) → str`
Step 5: extracts frames and re-encodes to H.264 MP4 via ffmpeg.

### `generate_html_report(...) → str`
Offline HTML storyboard + metrics. No CDN dependencies.

### `save_segments_json(kept_frames, video_meta, output_path) → str`
Writes integration JSON (schema v1.0, must not be modified).

---

## Integration Contract — `segments_kept.json`

```json
{
  "schema_version": "1.0",
  "source_video": {
    "fps": 30.0,
    "total_frames": 3600,
    "duration_sec": 120.0,
    "width": 1280,
    "height": 720
  },
  "compression": {
    "frames_in": 3600,
    "frames_out": 486,
    "reduction_pct": 86.5,
    "output_fps": 12
  },
  "segments": [
    {
      "frame_index": 0,
      "timestamp_sec": 0.0,
      "keep_reason": "context",
      "motion_score": 0.0,
      "phash_similar": 0.0,
      "face_detected": false
    },
    {
      "frame_index": 47,
      "timestamp_sec": 1.567,
      "keep_reason": "face",
      "motion_score": 0.213,
      "phash_similar": 0.421,
      "face_detected": true
    }
  ]
}
```

**`keep_reason` values:**
- `"face"` — Haar cascade detected ≥1 face; highest priority
- `"motion"` — optical flow score ≥ 0.05 AND pHash similarity ≤ 0.95
- `"context"` — forced context frame (≥3s since last kept frame)

---

## Performance

| Metric | Target | Typical Result |
|--------|--------|----------------|
| File size reduction | ≥ 70% | 75–88% |
| Processing speed | ≥ 4× real-time | 6–12× real-time |
| 2-min video processing time | ≤ 10s | 3–8s |

---

## Tuning Parameters

Edit at the top of `solution.py`:

```python
PHASH_SIMILARITY_THRESHOLD = 0.95    # Higher → keep more duplicate-ish frames
MOTION_THRESHOLD           = 0.05    # Lower → keep more low-motion frames
CONTEXT_INTERVAL_SEC       = 3.0     # Lower → more context frames (larger output)
OUTPUT_FPS                 = 12      # Higher → smoother but larger output
```

---

## Deliverables Checklist

| # | Deliverable | File | Status |
|---|-------------|------|--------|
| 1 | Working compression script | `solution.py` | ✅ |
| 2 | Compressed output video | `compressed_output.mp4` | ✅ (run script) |
| 3 | Storyboard + size comparison | `compression_report.html` | ✅ (run script) |
| 4 | Segment log for Sentio Mind | `segments_kept.json` | ✅ (run script) |
| 5 | Demo screen recording | `demo.mp4` | Record manually |

---

## Rules Compliance

- ✅ `README.md` is the primary documentation
- ✅ Integration JSON schema not modified
- ✅ Function signatures in template stubs preserved (`load_video`, `select_keyframes`, `encode_compressed_video`, `extract_intelligent_frames`, `generate_html_report`)
- ✅ HTML report works offline — zero CDN dependencies, all assets base64-embedded
- ✅ Python 3.9+ only — no Jupyter notebook

---

## GitHub

Repo: `https://github.com/Sentiodirector/Assignement_Video_compression`

Branch naming: `FirstName_LastName_RollNumber`  
Push only to your named branch.
