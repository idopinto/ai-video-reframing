"""
Smart Video Reframing — evaluation / research app.

Compare planners, inspect tracks, and tune hyperparameters.

    uv run streamlit run app.py

For the public demo, use demo.py instead.
"""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from smartcrop.detect import cache_path, face_cache_path, faces, tracks
from smartcrop.pipeline import reframe
from smartcrop.plan import DEFAULT_PLANNER, PLANNERS, Director, primary_track
from smartcrop.probe import catalogue, probe
from smartcrop.utils import (
    CACHE_DIR,
    OUTPUT_DIR,
    ROOT,
    TITLES,
    VIDEO_DIR,
    VIDEO_EXTENSIONS,
    atomic_write,
    center_x,
    crop_width,
    list_videos,
    portrait_name,
    x_bounds,
)

st.set_page_config(page_title="Smart Crop — Evaluation", layout="wide")

METHOD = PLANNERS[DEFAULT_PLANNER]
DETECTORS = ["yolo11n.pt", "yolo11m.pt"]
DETECTOR_FILTERS = {
    "yolo11n.pt": dict(min_track_conf=0.50, min_area=0.02, min_coverage=0.50),
}
TRACKERS = {
    "botsort + reid": str(ROOT / "smartcrop/trackers/botsort_reid.yaml"),
    "bytetrack": "bytetrack.yaml",
}
UPLOAD_TYPES = sorted(ext.lstrip(".") for ext in VIDEO_EXTENSIONS)

SUBJECT = (10, 214, 255)
SURVIVOR = (255, 176, 48)
FILTERED = (170, 170, 170)
FACE = (160, 64, 255)
WINDOWS = {"center": (48, 59, 255), "naive-track": (80, 220, 80)}
DIRECTOR_WINDOW = (0, 140, 255)


def video_dir_stamp() -> tuple:
    rows = []
    for path in list_videos(VIDEO_DIR):
        try:
            info = path.stat()
        except FileNotFoundError:
            continue
        rows.append((path.name, int(info.st_mtime), info.st_size))
    return tuple(rows)


@st.cache_data(show_spinner=False)
def clips(_stamp: tuple) -> pd.DataFrame:
    return catalogue()


@st.cache_data(show_spinner=False)
def clip_meta(path: str) -> dict:
    return probe(Path(path))


@st.cache_data(show_spinner=False)
def clip_tracks(video: str, model: str, tracker: str) -> tuple[pd.DataFrame, set[int]]:
    src = VIDEO_DIR / video
    meta = clip_meta(str(src))
    method = replace(
        METHOD, model=model, tracker=tracker, **DETECTOR_FILTERS.get(model, {})
    )
    dets = tracks(src, stride=method.stride, conf=method.conf, model=model, tracker=tracker)
    survivors = method.candidates(src, meta["width"] * meta["height"])
    return dets, set(survivors.track_id.astype(int))


def _label(frame: np.ndarray, text: str, at: tuple[int, int], colour, scale: float) -> None:
    for c, t in ((0, 0, 0), 5), (colour, 2):
        cv2.putText(frame, text, at, cv2.FONT_HERSHEY_SIMPLEX, scale, c, t, cv2.LINE_AA)


def annotated(
    video: str,
    windows: dict[str, np.ndarray],
    chosen: int | None,
    show_faces: bool,
    model: str,
    tracker: str,
) -> Path:
    src = VIDEO_DIR / video
    key = "|".join([video, *windows, str(chosen), str(show_faces), model, tracker])
    for x in windows.values():
        key += hashlib.md5(np.asarray(x).tobytes()).hexdigest()
    dst = CACHE_DIR / "annotated" / f"{src.stem}_{hashlib.md5(key.encode()).hexdigest()[:10]}.mp4"
    if dst.exists():
        return dst

    meta = clip_meta(str(src))
    w, h, fps = meta["width"], meta["height"], meta["fps"]
    cw = crop_width(h)
    dets, survivors = clip_tracks(video, model, tracker)
    by_frame = {f: g for f, g in dets.groupby("frame")}
    sampled = np.array(sorted(by_frame))
    found = (
        faces(src, METHOD.stride, METHOD.face_width, METHOD.face_min_score)
        if show_faces
        else None
    )
    faces_by_frame = {f: g for f, g in found.groupby("frame")} if found is not None else {}

    def draw(frame: np.ndarray, i: int) -> None:
        if len(sampled):
            nearest = sampled[np.abs(sampled - i).argmin()]
            here = by_frame[nearest].assign(kept=lambda f: f.track_id.isin(survivors))
            for r in here.sort_values("kept").itertuples():
                tid = -1 if pd.isna(r.track_id) else int(r.track_id)
                colour, thick = (
                    (SUBJECT, 5) if tid == chosen else (SURVIVOR, 3) if r.kept else (FILTERED, 1)
                )
                p1, p2 = (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2))
                cv2.rectangle(frame, p1, p2, colour, thick)
                if r.kept:
                    label = f"#{tid} {r.name} {r.conf:.2f}"
                    _label(frame, label, (p1[0] + 4, max(p1[1] - 6, 16)), colour, 0.55)
            for r in faces_by_frame.get(nearest, pd.DataFrame()).itertuples():
                cv2.rectangle(frame, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), FACE, 2)
        for k, (name, x) in enumerate(windows.items()):
            left = int(x[min(i, len(x) - 1)])
            colour = WINDOWS.get(name, DIRECTOR_WINDOW)
            cv2.rectangle(frame, (left, 0), (left + cw - 1, h - 1), colour, 3)
            _label(frame, name, (left + 8, h - 14 - 26 * k), colour, 0.7)

    def write(out: Path) -> None:
        enc = subprocess.Popen(
            [
                "ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
                "-c:v", "libx264", "-crf", "23", "-preset", "veryfast",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out),
            ],
            stdin=subprocess.PIPE,
        )
        cap = cv2.VideoCapture(str(src))
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            draw(frame, i)
            enc.stdin.write(frame.tobytes())
            i += 1
        cap.release()
        enc.stdin.close()
        if enc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed annotating {video}")

    return atomic_write(dst, write)


def output(video: str, planner, folder: str, force: bool = False):
    src = VIDEO_DIR / video
    if not src.exists():
        raise FileNotFoundError(
            f"{video} is no longer in data/videos/. Upload it again, or pick another clip."
        )
    dst = OUTPUT_DIR / folder / portrait_name(video)
    meta = clip_meta(str(src))
    if force or not dst.exists():
        return dst, reframe(src, dst, planner, meta)
    return dst, planner(src, meta)


@st.cache_data(show_spinner=False)
def video_bytes(path: str, mtime: float) -> bytes:
    return Path(path).read_bytes()


def offer_download(path: Path, filename: str, key: str, label: str = "Download") -> None:
    ready = path.exists() and path.stat().st_size > 0
    if not ready:
        st.button(label, disabled=True, width="stretch", key=f"{key}_disabled")
        return
    st.download_button(
        label=label,
        data=video_bytes(str(path), path.stat().st_mtime),
        file_name=filename,
        mime="video/mp4",
        key=key,
        width="stretch",
        icon=":material/download:",
        help="Save this video wherever you want.",
    )


@st.cache_data(show_spinner=False)
def titles() -> dict[str, str]:
    if not TITLES.exists():
        return {}
    return pd.read_csv(TITLES, dtype=str).set_index("file")["title"].to_dict()


def matches(file: str, title: str, query: str) -> bool:
    haystack = f"{file} {title}".lower()
    return all(word in haystack for word in query.lower().split())


def control_players(action: str) -> None:
    n = st.session_state.get("_player_cmd", 0) + 1
    st.session_state._player_cmd = n
    if action == "play":
        js = (
            "document.querySelectorAll('video').forEach(function(v,i){"
            "v.currentTime=0;if(i)v.muted=true;"
            "var p=v.play();if(p&&p.catch)p.catch(function(){v.muted=true;v.play();});"
            "});"
        )
    else:
        js = (
            "document.querySelectorAll('video').forEach(function(v){"
            "v.pause();v.currentTime=0;v.muted=false;});"
        )
    st.html(
        f"<script>/* {n} */{js}</script>",
        unsafe_allow_javascript=True,
        width="content",
    )


@st.cache_data(show_spinner="Deciding every clip with these settings…")
def decided_modes(folder: str, _planner: Director, _stamp: tuple) -> pd.Series:
    kinds = {}
    for f in clips(_stamp).file:
        src = VIDEO_DIR / f
        p = _planner
        if not cache_path(src, p.stride, p.conf, p.model, p.tracker).exists():
            kinds[f] = "not tracked yet"
            continue
        meta = clip_meta(str(src))
        if p.use_faces and not face_cache_path(
            src, p.stride, p.face_width, p.face_min_score
        ).exists():
            dets = p.candidates(
                src, meta["width"] * meta["height"], n_frames=meta["n_frames"]
            )
            if primary_track(dets, p.prefer) is not None:
                kinds[f] = "not tracked yet"
                continue
        d = p(src, meta)
        if d.mode == "tracking":
            kinds[f] = "tracking"
        elif d.x[0] == center_x(meta["width"], meta["height"]):
            kinds[f] = "stationary · centre"
        else:
            kinds[f] = "stationary · off centre"
    return pd.Series(kinds)


def save_upload(uploaded) -> Path:
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    dest = VIDEO_DIR / Path(uploaded.name).name
    dest.write_bytes(uploaded.getbuffer())
    clips.clear()
    clip_meta.clear()
    clip_tracks.clear()
    decided_modes.clear()
    return dest


st.sidebar.title("Smart Crop")
st.sidebar.caption("Evaluation app · compare planners and inspect tracks")

uploaded = st.sidebar.file_uploader(
    "Add a video",
    type=UPLOAD_TYPES,
    help="Saved into data/videos/. For large files, copy them there yourself.",
)
if uploaded is not None:
    marker = (uploaded.name, uploaded.size)
    if st.session_state.get("_uploaded") != marker:
        save_upload(uploaded)
        st.session_state._uploaded = marker
        st.sidebar.success(f"Saved {uploaded.name}")

stamp = video_dir_stamp()
df = clips(stamp)
if df.empty:
    st.sidebar.caption("No videos yet")
    st.title("Smart Crop evaluation")
    st.markdown(
        """
The **evaluation app** compares `center`, `naive-track`, and `director` on a local library.

1. Drop landscape videos into `data/videos/`, or use **Add a video** in the sidebar.
2. Pick methods in the sidebar and wait for the first render.
3. Download a portrait with the button under each result.

For a simpler public demo, run `uv run python demo.py`.
"""
    )
    st.stop()

st.sidebar.caption(f"{len(df)} videos · {df.duration.sum() / 60:.0f} min")

base = PLANNERS[DEFAULT_PLANNER]
default_tracker_label = next(
    label for label, path in TRACKERS.items() if path == base.tracker
)
model = st.session_state.get("director_detector", base.model)
if model not in DETECTORS:
    model = base.model
    st.session_state.director_detector = model
tracker_label = st.session_state.get("director_tracker", default_tracker_label)
if tracker_label not in TRACKERS:
    tracker_label = default_tracker_label
    st.session_state.director_tracker = tracker_label
steps = {
    "people": st.session_state.get("director_people", base.prefer is not None),
    "faces": st.session_state.get("director_faces", base.use_faces),
    "exit": st.session_state.get("director_exit", base.ease_exit),
    "smoothing": st.session_state.get("director_smoothing", base.smoothing),
}
tracker = TRACKERS[tracker_label]
director = replace(
    base,
    **DETECTOR_FILTERS.get(model, {}),
    model=model,
    tracker=tracker,
    prefer=base.prefer if steps["people"] else None,
    use_faces=steps["faces"],
    ease_exit=steps["exit"],
    smoothing=steps["smoothing"],
)
defaults = {
    "people": base.prefer is not None,
    "faces": base.use_faces,
    "exit": base.ease_exit,
    "smoothing": base.smoothing,
}
changed = "".join(
    f"-{step}" if on else f"-no-{step}" for step, on in steps.items() if on != defaults[step]
)
if tracker != base.tracker:
    changed = f"-{Path(tracker).stem}" + changed
if model != base.model:
    changed = f"-{Path(model).stem}" + changed
director_folder = DEFAULT_PLANNER + changed

decisions = decided_modes(director_folder, director, stamp)
counts = decisions.value_counts()
kinds = {
    "stationary · centre": counts.get("stationary · centre", 0),
    "stationary · off centre": counts.get("stationary · off centre", 0),
    "tracking": counts.get("tracking", 0),
}
if counts.get("not tracked yet", 0):
    kinds["not tracked yet"] = counts["not tracked yet"]

names = titles()
mode_keys = ["any", *kinds]
with st.sidebar.container(border=True, gap="small"):
    query = st.text_input(
        "Search",
        placeholder="filename, e.g. walk or street",
        type="search",
        label_visibility="collapsed",
        help="Matches the filename and optional title. Every word must appear.",
    )
    chosen_mode = st.pills(
        "Mode",
        mode_keys,
        default="any",
        format_func=lambda k: "any" if k == "any" else f"{k} ({kinds[k]})",
        help="How the director with the current settings decides each clip.",
    ) or "any"
    sel = df
    if query and query.strip():
        sel = sel[[matches(f, names.get(f, ""), query) for f in sel.file]]
    if chosen_mode != "any":
        sel = sel[sel.file.isin(decisions.index[decisions == chosen_mode])]
    if sel.empty:
        st.error("No videos match those filters.")
        st.stop()
    playlist = sel.file.tolist()
    if st.session_state.get("clip") not in playlist:
        st.session_state.clip = playlist[0]
    video = st.selectbox(
        f"{len(playlist)} videos",
        playlist,
        key="clip",
        format_func=lambda f: (
            f"{f} · {names[f]} · {df.loc[df.file == f, 'duration'].iat[0]:.0f}s"
            if names.get(f)
            else f"{f} · {df.loc[df.file == f, 'duration'].iat[0]:.0f}s"
        ),
    )
    methods = st.pills(
        "Methods",
        list(PLANNERS),
        selection_mode="multi",
        default=[DEFAULT_PLANNER],
        wrap=True,
    ) or []


def step(offset: int) -> None:
    i = playlist.index(st.session_state.clip)
    st.session_state.clip = playlist[(i + offset) % len(playlist)]


planners = {m: (PLANNERS[m], m) for m in methods}
if DEFAULT_PLANNER in methods:
    planners[DEFAULT_PLANNER] = (director, director_folder)
by_folder = {folder: planner for planner, folder in planners.values()}

with st.sidebar.expander("**director** settings", expanded=True):
    area = f"{director.min_area:g}"
    if director.prefer:
        area += ", not for a person"
    with st.popover("Hyperparameters", width="stretch"):
        st.table(
            pd.DataFrame(
                [
                    ("detector", Path(director.model).stem),
                    ("tracker", tracker_label),
                    ("stride", f"{director.stride:g}"),
                    ("confidence floor", f"{director.conf:g}"),
                    ("min coverage", f"{director.min_coverage:g}"),
                    ("min mean confidence", f"{director.min_track_conf:g}"),
                    ("min mean area", area),
                    ("prefer people", "on" if director.prefer else "off"),
                    ("use faces", "on" if director.use_faces else "off"),
                    ("face width", f"{director.face_width:g}"),
                    ("face min score", f"{director.face_min_score:g}"),
                    ("min face share", f"{director.min_face_share:g}"),
                    ("span threshold", f"{director.span_threshold:g}"),
                    ("snap distance", f"{director.snap_distance:g}"),
                    ("ease to centre", "on" if director.ease_exit else "off"),
                    ("recenter seconds", f"{director.recenter_seconds:g}"),
                    ("smoothing", "on" if director.smoothing else "off"),
                    ("sigma seconds", f"{director.sigma_seconds:.2g}"),
                ],
                columns=["parameter", "value"],
            ).set_index("parameter")
        )
    st.selectbox(
        "Detector",
        DETECTORS,
        index=DETECTORS.index(base.model),
        key="director_detector",
        help="YOLO11 weights. The public demo uses nano; this app defaults to medium.",
    )
    st.selectbox(
        "Tracker",
        list(TRACKERS),
        index=list(TRACKERS).index(default_tracker_label),
        key="director_tracker",
    )
    st.checkbox("Prefer people as subject", value=base.prefer is not None, key="director_people")
    st.checkbox("Frame the face", value=base.use_faces, key="director_faces")
    st.checkbox("Ease to centre on exit", value=base.ease_exit, key="director_exit")
    st.checkbox("Gaussian smoothing", value=base.smoothing, key="director_smoothing")

if st.sidebar.button("Re-render this clip", width="stretch"):
    for planner, folder in planners.values():
        output(video, planner, folder, force=True)
    st.sidebar.success("Re-rendered.")

st.sidebar.caption("Outputs are cached in `outputs/<method>/`.")

src = VIDEO_DIR / video
meta = clip_meta(str(src))
w, h = meta["width"], meta["height"]
cw = crop_width(h)
lo, hi = x_bounds(w, h)

if not methods:
    st.info("Pick at least one method in the sidebar.")
    st.stop()

with st.container(horizontal=True, horizontal_alignment="center"):
    st.button("Prev", icon=":material/chevron_left:", on_click=step, args=(-1,), key="prev_clip", width=180)
    st.button("Next", icon=":material/chevron_right:", on_click=step, args=(1,), key="next_clip", width=180)

with st.container(horizontal=True, horizontal_alignment="center"):
    play_all = st.button("Play all", icon=":material/play_arrow:", type="primary", key="play_all", width="content")
    reset_players = st.button("Reset to start", icon=":material/replay:", type="primary", key="reset_players", width="content")

with st.spinner("Rendering…"):
    try:
        results = {planners[m][1]: output(video, *planners[m]) for m in methods}
    except (FileNotFoundError, RuntimeError) as exc:
        st.error(str(exc))
        st.stop()

show_boxes = st.session_state.get("show_boxes", False)
cols = st.columns([3] + [1] * len(results))
with cols[0]:
    if show_boxes:
        director_runs = [
            (f, d) for f, (_, d) in results.items() if isinstance(by_folder[f], Director)
        ]
        chosen = director_runs[0][1].track_id if director_runs else None
        show_faces = any(by_folder[f].use_faces for f, _ in director_runs)
        box_model = by_folder[director_runs[0][0]].model if director_runs else METHOD.model
        box_tracker = by_folder[director_runs[0][0]].tracker if director_runs else METHOD.tracker
        with st.spinner("Drawing boxes…"):
            boxed = annotated(
                video,
                {f: d.x for f, (_, d) in results.items()},
                chosen,
                show_faces,
                box_model,
                box_tracker,
            )
        clip_title = names.get(video)
        head = f"{clip_title} · {video}" if clip_title else video
        caption = f"**{head}** · 🟨 subject · 🟦 kept · ⬜ filtered"
        caption += " · 🟪 face" if show_faces else ""
        st.caption(caption + " · crop windows labelled by method", text_alignment="center")
        st.video(str(boxed))
        offer_download(boxed, f"{Path(video).stem}_boxes.mp4", f"dl_boxes_{boxed.name}", "Download overlay")
    else:
        title = names.get(video)
        st.caption(f"{title} · {video}" if title else video, text_alignment="center")
        st.video(str(src))
    st.toggle(
        "Show boxes on the original",
        key="show_boxes",
        help="Draws crop windows, tracks, and faces on the source.",
    )
for col, (name, (dst, decision)) in zip(cols[1:], results.items()):
    with col:
        x = decision.x
        pos = int(x[0]) if np.all(x == x[0]) else f"{int(x.min())}–{int(x.max())}"
        st.caption(f"**{name}** · {cw}×{h} · x={pos}", text_alignment="center")
        st.video(str(dst))
        if isinstance(by_folder[name], Director):
            if decision.mode == "tracking":
                kind = "tracking"
            elif np.all(x == center_x(w, h)):
                kind = "stationary · centre"
            else:
                kind = "stationary · off centre"
            st.caption(f"**{kind}**", text_alignment="center")
        offer_download(dst, f"{Path(video).stem}_{name}.mp4", f"dl_{name}_{dst.name}")

if play_all:
    control_players("play")
elif reset_players:
    control_players("reset")

if st.toggle("Camera path x(t)", value=True):
    st.caption(f"Left edge of the crop. Valid range is [{lo}, {hi}].")
    n = meta["n_frames"]
    traj = pd.DataFrame(
        {name: decision.x[:n] for name, (_, decision) in results.items()},
        index=np.arange(n) / meta["fps"],
    )
    traj.index.name = "seconds"
    st.line_chart(traj, height=240)

with st.expander("Source properties"):
    st.dataframe(
        pd.Series({k: str(v) for k, v in meta.items()}).rename("value").to_frame(),
        width="stretch",
    )
