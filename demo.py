"""
Public Gradio demo for Smart Video Reframing.

Thin UI around smartcrop.pipeline.reframe. Built for phones first and Hugging Face
Spaces: short clips, YOLO11n, temporary files, Gradio webcam + upload.

Zero-cost host: a Gradio Space on ZeroGPU (free accounts get two). CPU Gradio
Spaces require Pro. `import spaces` must happen before torch.

    uv run python demo.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from dataclasses import replace
from pathlib import Path

try:
    import spaces
except ImportError:
    spaces = None

import gradio as gr
import numpy as np

from smartcrop.detect import ensure_face_weights, get_detector
from smartcrop.pipeline import reframe
from smartcrop.plan import Director
from smartcrop.probe import display_size, ensure_upright
from smartcrop.utils import ROOT, VIDEO_EXTENSIONS, center_x, crop_width

MAX_DURATION_S = 20
WARN_DURATION_S = 12
MAX_UPLOAD_MB = 100
DEMO_MODEL = "yolo11n.pt"
EXAMPLES_DIR = ROOT / "examples"

DEMO_PLANNER = replace(
    Director(),
    model=DEMO_MODEL,
    min_track_conf=0.50,
    min_area=0.02,
    min_coverage=0.50,
)

WORK = Path(tempfile.mkdtemp(prefix="smart-reframe-"))
_MODELS_READY = False
RUN_HINT = "uv run python demo.py"


def running_in_streamlit() -> bool:
    return Path(sys.argv[0]).name.lower() in {"streamlit", "streamlit.exe"}


if running_in_streamlit():
    import streamlit as st

    st.set_page_config(page_title="Smart Video Reframing", layout="centered")
    st.error("This demo is Gradio now. Streamlit cannot host it.")
    st.code(RUN_HINT, language="bash")
    st.caption("Then open the local Gradio URL (usually http://127.0.0.1:7860).")
    st.stop()

CSS = """
.gradio-container { max-width: 720px !important; margin: 0 auto !important; }
#reframe-btn, #reframe-btn button {
  width: 100% !important;
  min-height: 52px !important;
  font-size: 1.12rem !important;
}
#download-btn, #download-btn button {
  width: 100% !important;
  min-height: 48px !important;
}
#source-video video, #result-video video {
  width: 100% !important;
  max-width: 100% !important;
  height: auto !important;
}
#result-video video {
  max-height: 72vh;
  margin: 0 auto;
  background: #0b0d12;
}
"""


def ffmpeg_available() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        subprocess.run(["ffprobe", "-version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError, OSError):
        return False


def gpu(*, duration: int = 60):
    def deco(fn):
        if spaces is None:
            return fn
        return spaces.GPU(duration=duration)(fn)

    return deco


def load_models() -> None:
    global _MODELS_READY
    if _MODELS_READY:
        return
    device = "cuda" if spaces is not None else None
    get_detector(model=DEMO_MODEL, device=device)
    ensure_face_weights()
    _MODELS_READY = True


def list_examples() -> list[Path]:
    if not EXAMPLES_DIR.is_dir():
        return []
    return sorted(
        p
        for p in EXAMPLES_DIR.iterdir()
        if p.is_file()
        and p.suffix.lower() in VIDEO_EXTENSIONS
        and not p.stem.endswith("_portrait")
        and not p.name.startswith(".")
    )


def as_path(video) -> Path | None:
    if video is None:
        return None
    if isinstance(video, dict):
        video = video.get("path") or video.get("name")
    if not video:
        return None
    path = Path(str(video))
    return path if path.exists() else None


def framing_copy(decision, meta: dict) -> str:
    middle = center_x(meta["width"], meta["height"])
    if decision.mode == "tracking":
        if "easing to centre" in decision.reason:
            return "Followed the subject, then eased back when they left the frame."
        if "face" in decision.reason.lower():
            return "Followed the subject and framed on their face."
        return "Followed the subject as they moved."
    if "nothing to follow" in decision.reason.lower() or "no track" in decision.reason.lower():
        return "No clear subject — used a stable centre crop."
    if np.all(decision.x == middle):
        return "The subject barely moved, so the frame stayed locked."
    return "Held a stable window on an off-centre subject."


def validate_clip(src: Path, meta: dict) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    duration = float(meta.get("duration") or 0)
    width, height = display_size(meta)
    size_mb = src.stat().st_size / 1e6

    if duration > MAX_DURATION_S:
        errors.append(
            f"This clip is {duration:.1f}s. The hosted demo accepts videos up to "
            f"{MAX_DURATION_S}s. Trim it, or use the CLI / evaluation app locally."
        )
    elif duration >= WARN_DURATION_S:
        warnings.append(
            f"{duration:.0f}s is near the demo limit. Rendering can take a minute."
        )

    if size_mb > MAX_UPLOAD_MB:
        errors.append(
            f"File is {size_mb:.0f} MB. Please stay under {MAX_UPLOAD_MB} MB for this demo."
        )

    if crop_width(height) >= width:
        errors.append(
            "This clip is too tall for a full-height 9:16 crop. "
            "Hold the phone sideways and record landscape."
        )
    elif width / max(height, 1) < 1.2:
        warnings.append(
            "This looks closer to square or portrait. Results are best on 16:9 landscape."
        )

    return errors, warnings


def make_video_input() -> gr.Video:
    # Do not force 16:9 / 1280x720 — Safari on iPhone then records landscape
    # back-camera clips upside down. Prefer the rear camera; no selfie mirror.
    constraints = {"facingMode": {"ideal": "environment"}}
    kwargs = dict(
        label="Record or upload",
        sources=["webcam", "upload"],
        format="mp4",
        include_audio=True,
        elem_id="source-video",
    )
    if hasattr(gr, "WebcamOptions"):
        kwargs["webcam_options"] = gr.WebcamOptions(
            mirror=False,
            constraints=constraints,
        )
        kwargs["buttons"] = ["download"]
        try:
            return gr.Video(**kwargs)
        except TypeError:
            kwargs.pop("webcam_options", None)
            kwargs.pop("buttons", None)
    kwargs["webcam_constraints"] = constraints
    try:
        return gr.Video(**kwargs)
    except TypeError:
        kwargs.pop("webcam_constraints", None)
        return gr.Video(**kwargs)


@gpu(duration=60)
def reframe_clip(video):
    src = as_path(video)
    if src is None:
        raise gr.Error("Record or upload a short landscape video first.")
    if not ffmpeg_available():
        raise gr.Error("ffmpeg was not found. On Spaces, packages.txt should install it.")

    suffix = src.suffix.lower() if src.suffix.lower() in VIDEO_EXTENSIONS else ".mp4"
    work_src = WORK / f"source-{uuid.uuid4().hex[:8]}{suffix}"
    work_dst = WORK / f"portrait-{uuid.uuid4().hex[:8]}.mp4"
    shutil.copy2(src, work_src)

    try:
        work_src, meta = ensure_upright(work_src)
    except Exception as exc:
        raise gr.Error(f"Could not read this file. Try an H.264 MP4. ({exc})") from exc

    errors, warnings = validate_clip(work_src, meta)
    for warning in warnings:
        gr.Warning(warning)
    if errors:
        raise gr.Error(errors[0])

    try:
        load_models()
        decision = reframe(work_src, work_dst, DEMO_PLANNER, meta)
    except Exception as exc:
        raise gr.Error(f"Reframe failed. {exc}") from exc

    if not work_dst.exists():
        raise gr.Error("Reframe finished without writing a video. Try another clip.")

    caption = framing_copy(decision, meta)
    return str(work_dst), caption, str(work_dst)


if os.environ.get("SPACE_ID"):
    load_models()


with gr.Blocks(title="Smart Video Reframing") as demo:
    gr.Markdown(
        """
# Smart Video Reframing

Turn landscape video into vertical video  
while keeping the important subject in frame.
"""
    )
    if not ffmpeg_available():
        gr.Markdown(
            "**ffmpeg is missing.** On Hugging Face Spaces, `packages.txt` should install it. "
            "Locally: `brew install ffmpeg`."
        )

    source = make_video_input()
    gr.Markdown(
        "On iPhone: turn the phone sideways **before** you hit Record. "
        "If Safari still flips the clip, record in the Camera app and tap **Upload**."
    )
    reframe_btn = gr.Button("Reframe", variant="primary", elem_id="reframe-btn")
    status = gr.Markdown(visible=True, value="")
    result = gr.Video(
        label="Smart Crop",
        interactive=False,
        include_audio=True,
        elem_id="result-video",
    )

    download = gr.DownloadButton(
        "Download Result",
        elem_id="download-btn",
    )

    examples = list_examples()
    if examples:
        gr.Examples(
            examples=[[str(p)] for p in examples],
            inputs=[source],
            label="Try an example",
        )

    with gr.Accordion("Technical Details", open=False):
        gr.Markdown(
            f"""
The crop stays full height. The app only decides where to place the 9:16 window.

1. Find people and objects
2. Keep the reliable tracks
3. Choose the main subject (people first)
4. Aim at the face when we can
5. If they barely moved, hold still; otherwise follow them
6. If they leave, ease back to centre
7. Smooth the camera, then crop to 9:16

Hosted-demo limits: **{MAX_DURATION_S}s**, **{MAX_UPLOAD_MB} MB**, YOLO11n.  
Longer clips and YOLO11m live in the CLI (`python -m smartcrop`) and `app.py`.  
If the browser blocks camera recording, use **Upload**.
"""
        )

    reframe_btn.click(
        fn=reframe_clip,
        inputs=[source],
        outputs=[result, status, download],
        show_progress="full",
        concurrency_limit=1,
        api_name="reframe",
    )
    source.change(
        fn=lambda: (None, "", None),
        inputs=None,
        outputs=[result, status, download],
        show_progress="hidden",
        api_name=False,
    )


if __name__ == "__main__":
    queued = demo.queue(max_size=8)
    try:
        queued.launch(
            theme=gr.themes.Soft(primary_hue="amber"),
            css=CSS,
            ssr_mode=False,
        )
    except TypeError:
        queued.launch()
