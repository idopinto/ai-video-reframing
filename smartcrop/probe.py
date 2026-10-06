"""Read a clip's properties with ffprobe."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd

from smartcrop.utils import PROBE_CACHE, SRC_AR, VIDEO_DIR, crop_width, list_videos


def probe(path: Path) -> dict:
    """Video and audio properties of one file."""
    raw = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_streams",
            "-show_format",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    meta = json.loads(raw)

    video = next(s for s in meta["streams"] if s["codec_type"] == "video")
    audio = next((s for s in meta["streams"] if s["codec_type"] == "audio"), None)

    num, den = video["r_frame_rate"].split("/")
    fps = round(int(num) / int(den), 4) if int(den) else 0.0
    duration = round(float(meta["format"].get("duration") or 0.0), 3)

    n_frames = int(video.get("nb_frames") or 0)
    if n_frames <= 0 and fps > 0 and duration > 0:
        n_frames = int(round(duration * fps))

    rotation = _rotation_degrees(video)

    return {
        "file": path.name,
        "width": video["width"],
        "height": video["height"],
        "fps": fps,
        "duration": duration,
        "n_frames": n_frames,
        "vcodec": video["codec_name"],
        "pix_fmt": video["pix_fmt"],
        "sar": video.get("sample_aspect_ratio", "1:1"),
        "rotation": rotation,
        "acodec": audio["codec_name"] if audio else None,
        "a_ch": audio.get("channels") if audio else None,
        "a_rate": int(audio["sample_rate"]) if audio else None,
        "size_mb": round(path.stat().st_size / 1e6, 2),
    }


def catalogue(video_dir: Path = VIDEO_DIR, cache: Path = PROBE_CACHE) -> pd.DataFrame:
    """
    One row per clip. Raw values are cached to CSV. Derived columns are computed
    on load, so they cannot go stale. The cache is rebuilt if the directory
    contents change.
    """
    files = list_videos(video_dir)
    current = [p.name for p in files]

    if cache.exists() and cache.stat().st_size > 1:
        df = pd.read_csv(cache)
        cached = df["file"].astype(str).tolist() if "file" in df.columns else []
        if cached == current:
            return _with_derived(df)

    if not files:
        return pd.DataFrame()

    from tqdm import tqdm

    df = pd.DataFrame([probe(f) for f in tqdm(files, desc="ffprobe")])
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    return _with_derived(df)


def _with_derived(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["aspect"] = df["width"] / df["height"]
    df["is_16_9"] = (df["aspect"] - SRC_AR).abs() < 0.01
    df["has_audio"] = df["acodec"].notna()
    df["crop_w"] = df["height"].map(crop_width)
    df["x_max"] = df["width"] - df["crop_w"]
    df["kept_frac"] = df["crop_w"] / df["width"]
    df["center_x"] = df["x_max"] // 2
    df["mbps"] = df["size_mb"] * 8 / df["duration"].replace(0, pd.NA)
    return df


def _rotation_degrees(video: dict) -> int:
    rotation = 0.0
    tags = video.get("tags") or {}
    if tags.get("rotate") not in (None, ""):
        rotation = float(tags["rotate"])
    for side in video.get("side_data_list") or []:
        if side.get("rotation") is not None:
            rotation = float(side["rotation"])
    return int(round(rotation)) % 360


def display_size(meta: dict) -> tuple[int, int]:
    """Width and height as shown on screen, after rotation metadata."""
    width, height = int(meta["width"]), int(meta["height"])
    if int(meta.get("rotation") or 0) % 180 == 90:
        return height, width
    return width, height


def ensure_upright(src: Path) -> tuple[Path, dict]:
    """Re-encode so pixels match what you see; strip rotation tags."""
    src = Path(src)
    meta = probe(src)
    if int(meta.get("rotation") or 0) % 360 == 0:
        return src, meta

    dst = src.with_name(f"{src.stem}_upright.mp4")
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        str(src),
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-metadata:s:v:0",
        "rotate=0",
    ]
    audio = ["-c:a", "copy"] if meta.get("acodec") else ["-an"]
    result = subprocess.run(cmd + audio + [str(dst)], capture_output=True, text=True)
    if result.returncode != 0 and meta.get("acodec"):
        result = subprocess.run(
            cmd + ["-c:a", "aac", "-b:a", "128k", str(dst)],
            capture_output=True,
            text=True,
        )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip() or f"exit {result.returncode}"
        raise RuntimeError(f"ffmpeg failed to fix orientation: {detail}")
    return dst, probe(dst)
