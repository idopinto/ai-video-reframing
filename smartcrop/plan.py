"""
Plan where the 9:16 window sits in each frame.

Each planner takes a clip and its probed metadata and returns a `Decision`: the
crop's left edge `x` for every frame, plus the mode and reason behind it.

* `center`: a fixed centre crop.
* `naive_track`: the crudest detector-driven crop (class average, no instance).
* `Director`: the default method — one subject, hold or follow, then smooth.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from smartcrop.detect import DEFAULT_MODEL, TRACK_STRIDE, TRACKER, faces, tracks
from smartcrop.probe import probe
from smartcrop.utils import center_x, crop_width, x_bounds


def filter_tracks(
    dets: pd.DataFrame,
    frame_area: float,
    min_coverage: float,
    min_conf: float,
    min_area: float,
    prefer: frozenset[str] | None = None,
    *,
    n_sampled_frames: int,
) -> pd.DataFrame:
    """
    Keep tracks that could be the subject. Coverage is measured over all
    sampled frames in the clip, including frames with no tracked detections.

    Preferred classes skip the size floor: a distant person can still be the
    subject. Coverage and confidence still apply, so a sleeve at the frame
    edge (low conf) does not sneak through.
    """
    if dets.empty or dets.track_id.isna().all():
        return dets.iloc[:0]

    tracked = dets.dropna(subset=["track_id"])
    n_frames = max(n_sampled_frames, 1)

    stats = tracked.groupby("track_id").agg(
        coverage=("frame", lambda s: s.nunique() / n_frames),
        mean_conf=("conf", "mean"),
        mean_area=("area", "mean"),
        name=("name", lambda s: s.mode().iat[0]),
    )

    sized = stats.mean_area / frame_area >= min_area
    if prefer:
        sized = sized | stats.name.isin(prefer)
    keep = stats.index[
        (stats.coverage >= min_coverage) & (stats.mean_conf >= min_conf) & sized
    ]
    return tracked[tracked.track_id.isin(keep)]


def primary_subject(dets: pd.DataFrame) -> int | None:
    """The class with the highest summed score (conf × area). Used by `naive-track`."""
    if dets.empty:
        return None
    return int(dets.groupby("cls")["score"].sum().idxmax())


def primary_track(
    dets: pd.DataFrame, prefer: frozenset[str] | None = None
) -> int | None:
    """The track to follow for the whole clip: highest summed score (conf × area)."""
    if dets.empty or "track_id" not in dets or dets.track_id.isna().all():
        return None

    tracked = dets.dropna(subset=["track_id"])
    per_track = tracked.groupby("track_id").agg(
        score=("score", "sum"), name=("name", lambda s: s.mode().iat[0])
    )

    if prefer is not None and per_track.name.isin(prefer).any():
        per_track = per_track[per_track.name.isin(prefer)]

    return int(per_track.score.idxmax())


def face_centres(dets: pd.DataFrame, found: pd.DataFrame, track_id: int) -> pd.Series:
    """
    Face centre x of one person track, per frame it has a face.

    A face belongs to the smallest person box containing its centre, so with
    overlapping people it goes to the nearer, tighter box.
    """
    one = dets[dets.track_id == track_id]
    if one.empty or found.empty:
        return pd.Series(dtype=float)

    people = dets[dets.name == "person"].dropna(subset=["track_id"])
    pairs = found.merge(people, on="frame", suffixes=("", "_body"))
    inside = (
        (pairs.cx >= pairs.x1_body)
        & (pairs.cx <= pairs.x2_body)
        & (pairs.cy >= pairs.y1_body)
        & (pairs.cy <= pairs.y2_body)
    )
    pairs = pairs[inside].assign(body_area=lambda p: p.area)
    owner = pairs.sort_values("body_area").drop_duplicates(["frame", "x1", "y1"])
    mine = owner[owner.track_id == track_id]
    best = mine.sort_values("conf", ascending=False).drop_duplicates("frame")
    return best.set_index("frame")["cx"].sort_index().astype(float)


@dataclass
class Decision:
    """A camera path with the mode and reason behind it."""

    mode: str  # "stationary" | "tracking"
    x: np.ndarray
    span: float
    reason: str
    track_id: int | None = None


def center(src: Path, meta: dict) -> Decision:
    """Fixed centre crop."""
    n = max(int(meta["n_frames"]), 1)
    x = np.full(n, center_x(meta["width"], meta["height"]), dtype=int)
    return Decision("stationary", x, 0.0, "centre crop")


def naive_track(src: Path, meta: dict) -> Decision:
    """
    Follow the most common class's average box centre, every frame.

    No instance selection (two people pull the window into the gap between them),
    no mode decision, no smoothing. A reference for what the method avoids.
    """
    width, height = meta["width"], meta["height"]
    n = max(int(meta["n_frames"]), 1)

    dets = tracks(src)
    cls = primary_subject(dets)
    if cls is None:
        x = np.full(n, center_x(width, height), dtype=int)
        return Decision("stationary", x, 0.0, "nothing detected. Centre crop")

    prim = dets[dets.cls == cls]
    targets = prim.groupby("frame").apply(
        lambda g: np.average(g.cx, weights=g.score), include_groups=False
    )

    sampled = targets.index.to_numpy(dtype=float)
    centres = np.interp(np.arange(n, dtype=float), sampled, targets.to_numpy())

    left = centres - crop_width(height) / 2
    lo, hi = x_bounds(width, height)
    x = np.clip(np.rint(left), lo, hi).astype(int)
    span = float(targets.max() - targets.min()) / width
    return Decision("tracking", x, span, f"following class {prim.name.iat[0]!r}")


@dataclass(frozen=True)
class Director:
    """
    Follow one subject, and move the window only if they travel.

    1. Filter tracks too weak to be a subject. None left: hold centre.
    2. Pick one subject for the clip: people first, then summed ``conf × area``.
    3. Frame the face rather than the body box (`use_faces`).
    4. Mode: stationary (hold still) or tracking (follow).
    5. Ease back to centre if the subject leaves early (`ease_exit`).
    6. Gaussian-smooth a tracking path (`smoothing`).
    """

    model: str = DEFAULT_MODEL
    tracker: str = TRACKER
    stride: int = TRACK_STRIDE
    conf: float = 0.25

    min_coverage: float = 0.4
    min_track_conf: float = 0.60
    min_area: float = 0.05

    prefer: frozenset[str] = frozenset({"person"})

    use_faces: bool = True
    face_width: int = 640
    face_min_score: float = 0.6
    min_face_share: float = 0.3

    span_threshold: float = 0.15
    snap_distance: float = 0.05

    ease_exit: bool = True
    recenter_seconds: float = 2.0
    smoothing: bool = True
    sigma_seconds: float = 1 / 3

    def __call__(self, src: Path, meta: dict) -> Decision:
        return self.decide(src, meta)

    def decide(self, src: Path, meta: dict) -> Decision:
        width, height, fps = meta["width"], meta["height"], meta["fps"]
        n = max(int(meta["n_frames"]), 1)
        half = crop_width(height) / 2
        lo, hi = x_bounds(width, height)
        middle = center_x(width, height)

        def hold(value: int, span: float, reason: str, tid=None) -> Decision:
            return Decision(
                "stationary", np.full(n, value, dtype=int), span, reason, tid
            )

        dets = self.candidates(src, width * height, n_frames=n)
        chosen = primary_track(dets, self.prefer)
        if chosen is None:
            return hold(middle, 0.0, "no track survived filtering. Nothing to follow")

        points, face_note = self.framing_points(src, dets, chosen)
        frames = points.index.to_numpy(dtype=float)
        cx = points.to_numpy()

        span = float(cx.max() - cx.min()) / width
        if span <= self.span_threshold:
            settled = int(np.clip(round(float(np.median(cx)) - half), lo, hi))
            if abs(settled - middle) <= self.snap_distance * width:
                reason = f"span {span:.2f} <= {self.span_threshold}. Snapped to centre"
                return hold(middle, span, reason + face_note, chosen)
            reason = f"span {span:.2f} <= {self.span_threshold}. Holding at x={settled}"
            return hold(settled, span, reason + face_note, chosen)

        last = int(frames[-1])
        path = np.interp(np.arange(n, dtype=float), frames, cx)
        reason = f"span {span:.2f} > {self.span_threshold}. Following subject"

        if self.ease_exit and n - 1 - last > max(self.stride, round(0.2 * fps)):
            path = self.ease_to_centre(path, last, n, width, fps)
            reason += f". Subject leaves at {last / fps:.1f}s, easing to centre"

        x = np.clip(np.rint(path - half), lo, hi).astype(int)

        if self.smoothing:
            x = np.clip(np.rint(self.smooth(x, fps)), lo, hi).astype(int)
            reason += f". Gaussian blur, sigma {self.sigma_seconds:.2g}s"

        return Decision("tracking", x, span, reason + face_note, chosen)

    def candidates(
        self, src: Path, frame_area: float, *, n_frames: int | None = None
    ) -> pd.DataFrame:
        """Tracks that pass the filter."""
        if n_frames is None:
            n_frames = probe(src)["n_frames"]
        n_sampled_frames = n_frames // self.stride
        return filter_tracks(
            tracks(
                src,
                stride=self.stride,
                conf=self.conf,
                model=self.model,
                tracker=self.tracker,
            ),
            frame_area,
            self.min_coverage,
            self.min_track_conf,
            self.min_area,
            self.prefer,
            n_sampled_frames=n_sampled_frames,
        )

    def framing_points(
        self, src: Path, dets: pd.DataFrame, chosen: int
    ) -> tuple[pd.Series, str]:
        """
        Horizontal point to frame on, per sampled frame, and a note.

        Body centre plus the face's offset from it. The offset is measured where a
        face was found and interpolated elsewhere, so a missed face never makes
        the target jump between face and body. Faces in fewer than
        `min_face_share` of frames are ignored.
        """
        body = dets[dets.track_id == chosen].groupby("frame")["cx"].mean()
        if not self.use_faces:
            return body, ""
        found = faces(src, self.stride, self.face_width, self.face_min_score)
        face = face_centres(dets, found, chosen)
        face = face[face.index.isin(body.index)]
        share = len(face) / max(len(body), 1)

        if not len(face) or share < self.min_face_share:
            note = f". Body framing (face in only {share:.0%})" if share else ""
            return body, note

        offset = face - body.reindex(face.index)
        filled = np.interp(body.index, offset.index, offset.to_numpy())
        return body + filled, f". Framed on face ({share:.0%} of frames)"

    def ease_to_centre(
        self, path: np.ndarray, last: int, n: int, width: int, fps: float
    ) -> np.ndarray:
        """Glide to frame centre after the subject's last frame."""
        path = path.copy()
        tail = np.arange(last + 1, n)
        steps = max(int(round(self.recenter_seconds * fps)), 1)
        progress = np.clip((tail - last) / steps, 0.0, 1.0)
        eased = progress * progress * (3.0 - 2.0 * progress)
        path[tail] = path[last] + (width / 2.0 - path[last]) * eased
        return path

    def smooth(self, x: np.ndarray, fps: float) -> np.ndarray:
        """
        Gaussian blur, sigma in seconds converted per clip.

        The whole clip is known, so the average looks both ways and adds no lag.
        There is no hard speed cap: a fast subject still gets a fast window.
        """
        from scipy.ndimage import gaussian_filter1d

        return gaussian_filter1d(
            x.astype(float), self.sigma_seconds * fps, mode="nearest"
        )


Planner = Callable[[Path, dict], Decision]

DEFAULT_PLANNER = "director"

PLANNERS: dict[str, Planner] = {
    "center": center,
    "naive-track": naive_track,
    DEFAULT_PLANNER: Director(),
}
