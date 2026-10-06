"""Shared paths, 9:16 crop geometry, and atomic writes."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

VIDEO_DIR = ROOT / "data" / "videos"
OUTPUT_DIR = ROOT / "outputs"
CACHE_DIR = ROOT / "cache"
TITLES = ROOT / "data" / "titles.csv"

PROBE_CACHE = CACHE_DIR / "probe_cache.csv"
MODEL_DIR = CACHE_DIR / "models"
TRACK_DIR = CACHE_DIR / "tracks"
FACE_DIR = CACHE_DIR / "faces"

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".m4v", ".webm", ".avi"}

OUT_AR = 9 / 16
SRC_AR = 16 / 9


def list_videos(directory: Path = VIDEO_DIR) -> list[Path]:
    """Video files in `directory`, ignoring hidden files."""
    if not directory.is_dir():
        return []
    return sorted(
        p
        for p in directory.iterdir()
        if p.is_file()
        and not p.name.startswith(".")
        and p.suffix.lower() in VIDEO_EXTENSIONS
    )


def portrait_name(src: Path | str) -> str:
    """Output filename for a source clip. Always `.mp4`."""
    return f"{Path(src).stem}.mp4"


def crop_width(height: int) -> int:
    """
    Width of the 9:16 window, rounded down to even (H.264 needs even dimensions).

    Rounding down means a player that force-fits to 9:16 trims a little height
    rather than the left/right edges the crop chose.
    """
    return int(height * OUT_AR) // 2 * 2


def x_bounds(width: int, height: int) -> tuple[int, int]:
    """Inclusive range of the left edge: (0, width - crop_w)."""
    return 0, width - crop_width(height)


def clamp_x(x: int, width: int, height: int) -> int:
    lo, hi = x_bounds(width, height)
    return max(lo, min(int(x), hi))


def center_x(width: int, height: int) -> int:
    _, hi = x_bounds(width, height)
    return hi // 2


def atomic_write(dst: Path, write: Callable[[Path], None]) -> Path:
    """
    Write to a temporary sibling, then rename. An interrupted run never leaves a
    truncated file that a "reuse if it exists" check would trust.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    partial = dst.with_name(f".{dst.stem}.part{dst.suffix}")
    try:
        write(partial)
        partial.replace(dst)
    finally:
        partial.unlink(missing_ok=True)
    return dst
