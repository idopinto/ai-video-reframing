"""Cut the planned window out of the clip with ffmpeg."""

from __future__ import annotations

import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path

from smartcrop.utils import atomic_write, clamp_x, crop_width


def render(
    src: Path,
    dst: Path,
    x: int | Sequence[int],
    meta: dict,
    keep_audio: bool = True,
    crf: int = 18,
    preset: str = "medium",
) -> Path:
    """Crop `src` to 9:16 at `x` (one int, or one value per frame) into `dst`."""
    src = Path(src)
    if not src.exists():
        raise FileNotFoundError(f"input video not found: {src}")

    height, frame_w = meta["height"], meta["width"]
    width = crop_width(height)
    copy_audio = bool(keep_audio and meta["acodec"])

    xs = [clamp_x(v, frame_w, height) for v in ([x] if isinstance(x, int) else x)]
    if not xs:
        raise ValueError("x is empty")

    def command(out: Path, vf: str) -> list[str]:
        return [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(src),
            "-vf",
            vf,
            "-c:v",
            "libx264",
            "-crf",
            str(crf),
            "-preset",
            preset,
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            *(["-c:a", "copy"] if copy_audio else ["-an"]),
            str(out),
        ]

    if len(set(xs)) == 1:
        vf = f"crop={width}:{height}:{xs[0]}:0"
        return atomic_write(dst, lambda out: _run(command(out, vf)))

    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "crop.cmd"
        script.write_text(_sendcmd_script(xs, meta["fps"]))
        vf = f"sendcmd=f={_escape(script)},crop={width}:{height}:{xs[0]}:0"
        return atomic_write(dst, lambda out: _run(command(out, vf)))


def _run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        return
    detail = (result.stderr or result.stdout or "").strip()
    if not detail:
        detail = f"exit {result.returncode}"
    raise RuntimeError(f"ffmpeg failed: {detail}")


def _sendcmd_script(xs: Sequence[int], fps: float) -> str:
    """One sendcmd line per change in `xs`. Each line ends with a semicolon."""
    lines, previous = [], None
    for i, x in enumerate(xs):
        x = int(x)
        if x != previous:
            lines.append(f"{i / fps:.6f} crop x {x};")
            previous = x
    return "\n".join(lines) + "\n"


def _escape(path: Path) -> str:
    return str(path).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
