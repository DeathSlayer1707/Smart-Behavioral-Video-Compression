
import cv2
import numpy as np
import json
import os
import subprocess
import tempfile
import shutil
import time
import base64
import hashlib
from pathlib import Path
from PIL import Image
import imagehash
from typing import List, Dict, Tuple, Optional

# CONFIGURATION

PHASH_SIMILARITY_THRESHOLD = 0.95   # Drop if >95% similar to last kept frame
MOTION_THRESHOLD           = 0.05   # Discard if optical-flow score < this
CONTEXT_INTERVAL_SEC       = 3.0    # Force-keep one frame every N seconds
OUTPUT_FPS                 = 12     # Re-encode at 12 fps
HAAR_SCALE_FACTOR          = 1.1
HAAR_MIN_NEIGHBORS         = 4
HAAR_MIN_SIZE              = (30, 30)

HAAR_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# HELPER UTILITIES

def _frame_to_pil(frame_bgr: np.ndarray) -> Image.Image:
    """Convert OpenCV BGR frame to PIL RGB image."""
    return Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))


def _compute_phash(frame_bgr: np.ndarray) -> imagehash.ImageHash:
    """Compute perceptual hash of a frame."""
    pil_img = _frame_to_pil(frame_bgr)
    return imagehash.phash(pil_img)


def _phash_similarity(h1: imagehash.ImageHash, h2: imagehash.ImageHash) -> float:
    """
    Return similarity in [0, 1].
    hamming distance 0 → identical (1.0), max distance 64 → completely different (0.0).
    """
    max_bits = len(h1.hash) ** 2  # 64 for default pHash (8×8)
    distance = h1 - h2
    return 1.0 - (distance / max_bits)


def _optical_flow_score(prev_gray: np.ndarray, curr_gray: np.ndarray) -> float:
    """
    Compute mean magnitude of dense optical flow between two greyscale frames.
    Returns a float in [0, ∞); typical static scenes score < 0.05.
    """
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return float(np.mean(mag))


def _detect_faces(frame_bgr: np.ndarray, face_cascade: cv2.CascadeClassifier) -> bool:
    """Return True if at least one face is detected in the frame."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=HAAR_SCALE_FACTOR,
        minNeighbors=HAAR_MIN_NEIGHBORS,
        minSize=HAAR_MIN_SIZE
    )
    return len(faces) > 0


def _frame_to_base64_jpg(frame_bgr: np.ndarray, max_width: int = 320) -> str:
    """Encode a frame as a small base64 JPEG string (for HTML thumbnail)."""
    h, w = frame_bgr.shape[:2]
    if w > max_width:
        scale = max_width / w
        frame_bgr = cv2.resize(frame_bgr, (max_width, int(h * scale)))
    _, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 60])
    return base64.b64encode(buf).decode("utf-8")

# CORE ALGORITHM  — fixed function signatures


def load_video(video_path: str) -> Tuple[cv2.VideoCapture, dict]:
    """
    Open a video file and return (cap, meta).

    meta keys:
        fps         – frames per second (float)
        total_frames – total frame count (int)
        width, height – frame dimensions
        duration_sec  – video duration in seconds
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = total_frames / fps

    meta = {
        "fps": fps,
        "total_frames": total_frames,
        "width": width,
        "height": height,
        "duration_sec": round(duration_sec, 3),
    }
    return cap, meta


def compute_perceptual_hash(frame: np.ndarray) -> imagehash.ImageHash:
    """
    Step 1 helper — return pHash for a single BGR frame.
    Public wrapper so the pipeline can call it frame-by-frame.
    """
    return _compute_phash(frame)


def compute_optical_flow(prev_frame: np.ndarray, curr_frame: np.ndarray) -> float:
    """
    Step 2 helper — return mean optical-flow magnitude between two BGR frames.
    """
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    return _optical_flow_score(prev_gray, curr_gray)


def detect_faces_haar(frame: np.ndarray) -> bool:
    """
    Step 3 helper — return True if any face found in a BGR frame.
    Loads the cascade on first call and caches it.
    """
    if not hasattr(detect_faces_haar, "_cascade"):
        detect_faces_haar._cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
    return _detect_faces(frame, detect_faces_haar._cascade)


def select_keyframes(
    video_path: str,
    phash_threshold: float = PHASH_SIMILARITY_THRESHOLD,
    motion_threshold: float = MOTION_THRESHOLD,
    context_interval_sec: float = CONTEXT_INTERVAL_SEC,
) -> List[Dict]:
    """
    Main frame-selection pass.

    Returns a list of dicts — one per KEPT frame:
        {
            "frame_index": int,       # 0-based index in source video
            "timestamp_sec": float,   # time in seconds
            "keep_reason": str,       # "face" | "motion" | "context"
            "motion_score": float,
            "phash_similar": float,   # similarity to last-kept frame (0-1)
            "face_detected": bool,
        }
    """
    cap, meta = load_video(video_path)
    fps = meta["fps"]
    face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)

    kept_frames: List[Dict] = []

    last_kept_hash: Optional[imagehash.ImageHash] = None
    last_kept_frame: Optional[np.ndarray] = None
    last_context_ts: float = -context_interval_sec  # force first context frame

    frame_idx = 0
    prev_gray: Optional[np.ndarray] = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_sec = frame_idx / fps

        # Step 1: Perceptual hash similarity 
        curr_hash = _compute_phash(frame)
        phash_sim = 0.0
        if last_kept_hash is not None:
            phash_sim = _phash_similarity(curr_hash, last_kept_hash)
            if phash_sim > phash_threshold:
                # Nearly identical to last kept frame — skip (unless face/context)
                # We will still check face and context below before final discard
                pass

        #  Step 2: Optical flow motion score 
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        motion_score = 0.0
        if prev_gray is not None:
            motion_score = _optical_flow_score(prev_gray, curr_gray)

        # Step 3: Haar face detection 
        face_detected = _detect_faces(frame, face_cascade)

        #  Decision logic 
        keep_reason: Optional[str] = None

        if face_detected:
            keep_reason = "face"
        elif phash_sim <= phash_threshold and motion_score >= motion_threshold:
            keep_reason = "motion"

        #  Step 4: Context frame every 3 seconds 
        if keep_reason is None:
            if (timestamp_sec - last_context_ts) >= context_interval_sec:
                keep_reason = "context"

        if keep_reason is not None:
            kept_frames.append({
                "frame_index":   frame_idx,
                "timestamp_sec": round(timestamp_sec, 4),
                "keep_reason":   keep_reason,
                "motion_score":  round(motion_score, 6),
                "phash_similar": round(phash_sim, 4),
                "face_detected": face_detected,
            })
            last_kept_hash  = curr_hash
            last_kept_frame = frame
            if keep_reason == "context":
                last_context_ts = timestamp_sec

        prev_gray = curr_gray
        frame_idx += 1

    cap.release()
    return kept_frames


def encode_compressed_video(
    video_path: str,
    kept_frames: List[Dict],
    output_path: str,
    output_fps: int = OUTPUT_FPS,
) -> str:
    """
    Step 5 — extract surviving frames and re-encode to H.264 MP4 at output_fps.

    Strategy:
        1. Dump kept frames as sequentially numbered PNGs in a temp dir.
        2. Pipe them through ffmpeg with libx264 + CRF 23.

    Returns the output_path on success.
    """
    cap, meta = load_video(video_path)
    frame_index_set = {f["frame_index"] for f in kept_frames}

    tmp_dir = tempfile.mkdtemp(prefix="sentio_frames_")
    try:
        #  Extract frames 
        seq = 0
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx in frame_index_set:
                out_name = os.path.join(tmp_dir, f"frame_{seq:06d}.png")
                cv2.imwrite(out_name, frame)
                seq += 1
            idx += 1
        cap.release()

        if seq == 0:
            raise RuntimeError("No frames survived filtering — cannot encode output.")

        #  ffmpeg encode 
        pattern = os.path.join(tmp_dir, "frame_%06d.png")
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(output_fps),
            "-i", pattern,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return output_path


def save_segments_json(
    kept_frames: List[Dict],
    video_meta: dict,
    output_path: str,
) -> str:
    """
    Produce segments_kept.json with the exact integration schema.

    Schema (DO NOT MODIFY):
    {
      "schema_version": "1.0",
      "source_video": { "fps": float, "total_frames": int,
                        "duration_sec": float, "width": int, "height": int },
      "compression": { "frames_in": int, "frames_out": int,
                       "reduction_pct": float, "output_fps": int },
      "segments": [
        {
          "frame_index": int,
          "timestamp_sec": float,
          "keep_reason": "face|motion|context",
          "motion_score": float,
          "phash_similar": float,
          "face_detected": bool
        },
        ...
      ]
    }
    """
    frames_in  = video_meta["total_frames"]
    frames_out = len(kept_frames)
    reduction  = round(100.0 * (1 - frames_out / max(frames_in, 1)), 2)

    payload = {
        "schema_version": "1.0",
        "source_video": {
            "fps":           video_meta["fps"],
            "total_frames":  video_meta["total_frames"],
            "duration_sec":  video_meta["duration_sec"],
            "width":         video_meta["width"],
            "height":        video_meta["height"],
        },
        "compression": {
            "frames_in":    frames_in,
            "frames_out":   frames_out,
            "reduction_pct": reduction,
            "output_fps":   OUTPUT_FPS,
        },
        "segments": kept_frames,
    }

    with open(output_path, "w") as fh:
        json.dump(payload, fh, indent=2)

    return output_path


def generate_html_report(
    video_path: str,
    output_video_path: str,
    kept_frames: List[Dict],
    video_meta: dict,
    segments_json_path: str,
    report_path: str,
) -> str:
    """
    Generate an offline-capable HTML storyboard + size comparison report.
    All thumbnails are embedded as base64 data URIs.
    No external CDN dependencies.
    """
    #  Size stats 
    orig_size   = os.path.getsize(video_path)
    comp_size   = os.path.getsize(output_video_path) if os.path.exists(output_video_path) else 0
    reduction   = round(100.0 * (1 - comp_size / max(orig_size, 1)), 1)
    frames_in   = video_meta["total_frames"]
    frames_out  = len(kept_frames)

    reason_counts = {"face": 0, "motion": 0, "context": 0}
    for f in kept_frames:
        reason_counts[f["keep_reason"]] = reason_counts.get(f["keep_reason"], 0) + 1

    #  Storyboard thumbnails (every N kept frames) 
    thumb_every = max(1, len(kept_frames) // 30)
    thumb_indices = [f["frame_index"] for i, f in enumerate(kept_frames) if i % thumb_every == 0]
    thumb_index_set = set(thumb_indices)
    thumb_map: Dict[int, str] = {}

    cap = cv2.VideoCapture(video_path)
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx in thumb_index_set:
            thumb_map[idx] = _frame_to_base64_jpg(frame, max_width=160)
        idx += 1
    cap.release()

    #  Build storyboard HTML 
    storyboard_html = ""
    for fidx in thumb_indices:
        b64 = thumb_map.get(fidx, "")
        meta_f = next((f for f in kept_frames if f["frame_index"] == fidx), {})
        ts  = meta_f.get("timestamp_sec", 0)
        rsn = meta_f.get("keep_reason", "?")
        fdet = "✔" if meta_f.get("face_detected") else "✘"
        color = {"face": "#e74c3c", "motion": "#2980b9", "context": "#27ae60"}.get(rsn, "#888")
        storyboard_html += f"""
        <div class="thumb-card">
          <img src="data:image/jpeg;base64,{b64}" alt="frame {fidx}">
          <div class="thumb-meta">
            <span style="color:{color};font-weight:bold">{rsn.upper()}</span>
            <span>t={ts:.2f}s</span>
            <span>face:{fdet}</span>
          </div>
        </div>"""

    #  Motion score chart data 
    chart_labels = json.dumps([f["timestamp_sec"] for f in kept_frames[::max(1, len(kept_frames)//200)]])
    chart_data   = json.dumps([f["motion_score"]  for f in kept_frames[::max(1, len(kept_frames)//200)]])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>SentioMind Video Compression Report</title>
<style>
  :root {{
    --bg: #0f172a; --card: #1e293b; --accent: #0ea5e9;
    --green: #22c55e; --red: #ef4444; --text: #e2e8f0; --muted: #94a3b8;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text); font-family: 'Segoe UI', sans-serif; padding: 24px; }}
  h1   {{ font-size: 1.8rem; color: var(--accent); margin-bottom: 4px; }}
  h2   {{ font-size: 1.1rem; color: var(--accent); margin: 20px 0 10px; }}
  .subtitle {{ color: var(--muted); font-size: 0.85rem; margin-bottom: 24px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 24px; }}
  .stat-card {{ background: var(--card); border-radius: 10px; padding: 20px; text-align: center; }}
  .stat-card .value {{ font-size: 2.2rem; font-weight: 700; }}
  .stat-card .label {{ color: var(--muted); font-size: 0.78rem; margin-top: 4px; }}
  .green {{ color: var(--green); }} .red {{ color: var(--red); }} .blue {{ color: var(--accent); }}
  .bar-wrap {{ background: #334155; border-radius: 6px; height: 22px; overflow: hidden; margin: 8px 0; }}
  .bar {{ height: 100%; border-radius: 6px; display: flex; align-items: center; justify-content: flex-end; padding-right: 8px; font-size: 0.75rem; }}
  .section {{ background: var(--card); border-radius: 10px; padding: 20px; margin-bottom: 20px; }}
  .storyboard {{ display: flex; flex-wrap: wrap; gap: 8px; }}
  .thumb-card {{ background: #0f172a; border-radius: 6px; overflow: hidden; width: 160px; flex-shrink: 0; }}
  .thumb-card img {{ width: 100%; display: block; }}
  .thumb-meta {{ padding: 4px 6px; font-size: 0.67rem; display: flex; gap: 6px; flex-wrap: wrap; color: var(--muted); }}
  canvas {{ width: 100% !important; max-width: 900px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.78rem; }}
  th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid #334155; }}
  th {{ color: var(--accent); }}
  tr:hover td {{ background: #0f172a44; }}
</style>
</head>
<body>

<h1>🎬 SentioMind Smart Compression Report</h1>
<p class="subtitle">Behavioural video compression · pHash + Optical Flow + Haar Face Detection</p>

<!-- ── Stat Cards ── -->
<div class="grid">
  <div class="stat-card">
    <div class="value green">{reduction}%</div>
    <div class="label">File Size Reduction</div>
  </div>
  <div class="stat-card">
    <div class="value blue">{_human_bytes(orig_size)}</div>
    <div class="label">Original Size</div>
  </div>
  <div class="stat-card">
    <div class="value blue">{_human_bytes(comp_size)}</div>
    <div class="label">Compressed Size</div>
  </div>
  <div class="stat-card">
    <div class="value">{frames_in:,}</div>
    <div class="label">Total Frames In</div>
  </div>
  <div class="stat-card">
    <div class="value green">{frames_out:,}</div>
    <div class="label">Frames Kept</div>
  </div>
  <div class="stat-card">
    <div class="value">{video_meta['duration_sec']:.1f}s</div>
    <div class="label">Source Duration</div>
  </div>
</div>

<!-- ── Frame Retention Breakdown ── -->
<div class="section">
  <h2>Frame Retention Breakdown</h2>
  <p style="color:var(--muted);font-size:0.82rem;margin-bottom:12px">
    {frames_out} / {frames_in} frames kept &nbsp;·&nbsp;
    {reason_counts['face']} by face &nbsp;·&nbsp;
    {reason_counts['motion']} by motion &nbsp;·&nbsp;
    {reason_counts['context']} by context
  </p>
  <div style="max-width:500px">
    {"".join([
        _bar_html(label, count, frames_out, color)
        for label, count, color in [
            ("Face",    reason_counts['face'],    "#ef4444"),
            ("Motion",  reason_counts['motion'],  "#0ea5e9"),
            ("Context", reason_counts['context'], "#22c55e"),
        ]
    ])}
  </div>
</div>

<!-- ── Size Comparison Bar ── -->
<div class="section">
  <h2>Size Comparison</h2>
  <p style="color:var(--muted);font-size:0.82rem;margin-bottom:10px">Original vs Compressed</p>
  <div style="max-width:500px">
    {_bar_html("Original",    orig_size, orig_size, "#ef4444", _human_bytes(orig_size))}
    {_bar_html("Compressed",  comp_size, orig_size, "#22c55e", _human_bytes(comp_size))}
  </div>
</div>

<!-- ── Motion Score Chart ── -->
<div class="section">
  <h2>Motion Score Over Time (kept frames)</h2>
  <canvas id="motionChart" height="80"></canvas>
</div>

<!-- ── Storyboard ── -->
<div class="section">
  <h2>Storyboard (sampled kept frames)</h2>
  <div class="storyboard">
    {storyboard_html}
  </div>
</div>

<!-- ── Per-frame Table ── -->
<div class="section">
  <h2>Full Frame Log</h2>
  <div style="max-height:360px;overflow-y:auto">
  <table>
    <thead><tr>
      <th>#</th><th>Time (s)</th><th>Reason</th>
      <th>Motion</th><th>pHash Sim</th><th>Face</th>
    </tr></thead>
    <tbody>
    {"".join([
        f'<tr>'
        f'<td>{f["frame_index"]}</td>'
        f'<td>{f["timestamp_sec"]}</td>'
        f'<td style="color:{ {"face":"#ef4444","motion":"#0ea5e9","context":"#22c55e"}.get(f["keep_reason"],"#888") }">'
        f'{f["keep_reason"].upper()}</td>'
        f'<td>{f["motion_score"]:.4f}</td>'
        f'<td>{f["phash_similar"]:.3f}</td>'
        f'<td>{"✔" if f["face_detected"] else "✘"}</td>'
        f'</tr>'
        for f in kept_frames
    ])}
    </tbody>
  </table>
  </div>
</div>

<!-- ── Inline Chart.js (no CDN) ── -->
<script>
// Minimal Chart.js-compatible inline bar/line drawing using Canvas 2D API
(function() {{
  var canvas = document.getElementById('motionChart');
  var ctx = canvas.getContext('2d');
  canvas.width = canvas.parentElement.offsetWidth || 800;
  canvas.height = 180;
  var labels = {chart_labels};
  var data   = {chart_data};
  if (!data.length) return;
  var maxVal = Math.max(...data, MOTION_THRESHOLD_JS = 0.05) * 1.1;
  var W = canvas.width, H = canvas.height;
  var padL = 50, padR = 10, padT = 10, padB = 30;
  var plotW = W - padL - padR, plotH = H - padT - padB;
  ctx.fillStyle = '#1e293b'; ctx.fillRect(0, 0, W, H);
  // grid
  ctx.strokeStyle = '#334155'; ctx.lineWidth = 1;
  for (var i = 0; i <= 4; i++) {{
    var y = padT + plotH - (i / 4) * plotH;
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(padL + plotW, y); ctx.stroke();
    ctx.fillStyle = '#64748b'; ctx.font = '10px sans-serif';
    ctx.fillText((maxVal * i / 4).toFixed(3), 2, y + 4);
  }}
  // threshold line
  var ty = padT + plotH - (0.05 / maxVal) * plotH;
  ctx.strokeStyle = '#f59e0b'; ctx.lineWidth = 1.5; ctx.setLineDash([4,4]);
  ctx.beginPath(); ctx.moveTo(padL, ty); ctx.lineTo(padL + plotW, ty); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = '#f59e0b'; ctx.fillText('threshold 0.05', padL + 4, ty - 4);
  // data line
  ctx.strokeStyle = '#0ea5e9'; ctx.lineWidth = 2;
  ctx.beginPath();
  data.forEach(function(v, i) {{
    var x = padL + (i / (data.length - 1)) * plotW;
    var y2 = padT + plotH - (v / maxVal) * plotH;
    i === 0 ? ctx.moveTo(x, y2) : ctx.lineTo(x, y2);
  }});
  ctx.stroke();
  // x-axis label
  ctx.fillStyle = '#94a3b8'; ctx.font = '10px sans-serif';
  ctx.fillText('Time (seconds)', padL + plotW / 2 - 30, H - 4);
}})();
</script>

</body>
</html>
"""

    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(html)

    return report_path

# HTML HELPERS  (used only inside generate_html_report)

def _human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _bar_html(label: str, value: float, total: float, color: str, display: str = "") -> str:
    pct = round(100 * value / max(total, 1))
    show = display or f"{pct}%"
    return (
        f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">'
        f'  <span style="width:80px;font-size:0.78rem;color:#94a3b8">{label}</span>'
        f'  <div class="bar-wrap" style="flex:1">'
        f'    <div class="bar" style="width:{pct}%;background:{color}">{show}</div>'
        f'  </div>'
        f'</div>'
    )

# INTEGRATION ENTRY POINT

def extract_intelligent_frames(segments_json_path: str, video_path: str) -> List[np.ndarray]:
    """
    Integration contract function.
    Reads segments_kept.json and returns the list of selected BGR frames
    without re-running the full pipeline — used by the main Sentio pipeline.

    This replaces the full scan of the raw video.
    """
    with open(segments_json_path) as fh:
        data = json.load(fh)

    keep_indices = {seg["frame_index"] for seg in data["segments"]}
    cap, _ = load_video(video_path)

    frames: List[np.ndarray] = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx in keep_indices:
            frames.append(frame)
        idx += 1
    cap.release()
    return frames

# MAIN PIPELINE

def compress_video(
    input_path:     str = "video_sample_1.mov",
    output_video:   str = "compressed_output.mp4",
    output_json:    str = "segments_kept.json",
    output_html:    str = "compression_report.html",
) -> dict:
    """
    Full end-to-end compression pipeline.
    Returns a summary dict with timing and size stats.
    """
    print(f"[SentioMind] Loading video: {input_path}")
    t_start = time.perf_counter()

    _, meta = load_video(input_path)
    print(f"  → {meta['total_frames']} frames · {meta['fps']:.2f} fps · "
          f"{meta['width']}×{meta['height']} · {meta['duration_sec']:.1f}s")

    print("[SentioMind] Selecting key frames …")
    t1 = time.perf_counter()
    kept = select_keyframes(input_path)
    t2 = time.perf_counter()
    sel_time = t2 - t1
    print(f"  → {len(kept)} / {meta['total_frames']} frames kept  ({sel_time:.2f}s)")

    print("[SentioMind] Encoding compressed video …")
    encode_compressed_video(input_path, kept, output_video)
    print(f"  → written: {output_video}")

    #  JSON Output for integration
    save_segments_json(kept, meta, output_json)
    print(f"  → written: {output_json}")

    # HTML report 
    generate_html_report(input_path, output_video, kept, meta, output_json, output_html)
    print(f"  → written: {output_html}")

    #  Summary 
    t_end = time.perf_counter()
    total_time = t_end - t_start
    orig_size  = os.path.getsize(input_path)
    comp_size  = os.path.getsize(output_video) if os.path.exists(output_video) else 0
    reduction  = round(100 * (1 - comp_size / max(orig_size, 1)), 1)
    realtime_x = round(meta["duration_sec"] / max(total_time, 0.001), 1)

    summary = {
        "total_time_sec":  round(total_time, 2),
        "realtime_speed":  f"{realtime_x}×",
        "frames_in":       meta["total_frames"],
        "frames_kept":     len(kept),
        "original_bytes":  orig_size,
        "compressed_bytes": comp_size,
        "reduction_pct":   reduction,
        "output_fps":      OUTPUT_FPS,
    }

    print("\n")
    print(f"  Reduction     : {reduction}%")
    print(f"  Original size : {_human_bytes(orig_size)}")
    print(f"  Output size   : {_human_bytes(comp_size)}")
    print(f"  Total time    : {total_time:.2f}s  ({realtime_x}× real-time)")
    print("\n")
    return summary


if __name__ == "__main__":
    import sys
    input_video = sys.argv[1] if len(sys.argv) > 1 else "video_sample_1.mov"
    compress_video(input_path=input_video)
