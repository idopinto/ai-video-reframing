"""
Run the models. Tracks (YOLO11m + BoT-SORT with ReID) and faces (YuNet), cached
per clip. What they mean for the crop is decided in `plan.py`.

Both read each clip sequentially. Seeking to an arbitrary frame often means
decoding from the previous keyframe, so a full sequential pass is cheaper on
typical web and mobile encodings.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

from smartcrop.utils import FACE_DIR, MODEL_DIR, ROOT, TRACK_DIR, atomic_write

# Medium model: nano can misread scenery as subjects (for example a kiosk as a bus).
DEFAULT_MODEL = "yolo11m.pt"
TRACK_STRIDE = 3  # larger gaps between tracked frames mean more ID switches
# BoT-SORT with ReID compares appearance, so people who cross keep their ids.
# ByteTrack (motion and overlap only) is more likely to swap them.
TRACKER = str(ROOT / "smartcrop" / "trackers" / "botsort_reid.yaml")

COLUMNS = [
    "frame",
    "time",
    "track_id",
    "cls",
    "name",
    "conf",
    "x1",
    "y1",
    "x2",
    "y2",
    "cx",
    "cy",
    "w",
    "h",
    "area",
    "score",
]

# Enforced on read and write: an all-null track_id otherwise round-trips as
# float64 and comparisons silently stop matching.
DTYPES = {
    "frame": "int64",
    "time": "float64",
    "track_id": "Int64",
    "cls": "int64",
    "name": "string",
    "conf": "float64",
    "x1": "float64",
    "y1": "float64",
    "x2": "float64",
    "y2": "float64",
    "cx": "float64",
    "cy": "float64",
    "w": "float64",
    "h": "float64",
    "area": "float64",
    "score": "float64",
}


def best_device() -> str:
    """GPU when it is safe to use one.

    PyTorch's Metal backend can segfault if a tensor is moved there from a
    background thread (MetalShaderLibrary, during ``tensor.float()``).
    Streamlit runs the page on ``ScriptRunner.scriptThread``, so the app
    stays on CPU. The CLI runs on the main thread and still gets MPS.
    """
    import threading

    import torch

    on_main = threading.current_thread() is threading.main_thread()
    if torch.backends.mps.is_available() and on_main:
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class Detector:
    """YOLO + tracker over a video, one row per detection."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        device: str | None = None,
        conf: float = 0.25,
        classes: list[int] | None = None,
    ):
        from ultralytics import YOLO

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        weights = MODEL_DIR / model
        if not weights.exists():
            YOLO(model)
            downloaded = Path.cwd() / model
            if downloaded.exists():
                downloaded.replace(weights)
        self.model = YOLO(str(weights) if weights.exists() else model)
        self.device = device or best_device()
        self.conf = conf
        self.classes = classes
        self.names: dict[int, str] = self.model.names

    def track(
        self,
        src: Path,
        stride: int = TRACK_STRIDE,
        fps: float | None = None,
        verbose: bool = False,
        tracker: str = TRACKER,
    ) -> pd.DataFrame:
        rows: list[dict] = []
        results = self.model.track(
            source=str(src),
            stream=True,
            vid_stride=stride,
            conf=self.conf,
            classes=self.classes,
            device=self.device,
            tracker=tracker,
            persist=False,
            verbose=verbose,
        )
        # Ultralytics grabs `stride` frames and decodes the last one, so sample
        # i is really frame i * stride + stride - 1, not i * stride.
        for i, result in enumerate(results):
            self._collect(rows, result, frame=i * stride + stride - 1, fps=fps)
        return pd.DataFrame(rows, columns=COLUMNS)

    def _collect(self, rows: list, result, frame: int, fps: float | None) -> None:
        boxes = result.boxes
        if boxes is None or not len(boxes):
            return

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        ids = (
            boxes.id.cpu().numpy().astype(int)
            if boxes.id is not None
            else [None] * len(confs)
        )

        for (x1, y1, x2, y2), c, k, tid in zip(xyxy, confs, clss, ids):
            w, h = float(x2 - x1), float(y2 - y1)
            area = w * h
            rows.append(
                {
                    "frame": frame,
                    "time": frame / fps if fps else float("nan"),
                    "track_id": None if tid is None else int(tid),
                    "cls": int(k),
                    "name": self.names[int(k)],
                    "conf": float(c),
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                    "cx": float(x1) + w / 2,
                    "cy": float(y1) + h / 2,
                    "w": w,
                    "h": h,
                    "area": area,
                    "score": float(c) * area,
                }
            )


@lru_cache(maxsize=4)
def get_detector(
    model: str = DEFAULT_MODEL, conf: float = 0.25, device: str | None = None
) -> Detector:
    """One detector per (model, conf), so weights load once."""
    return Detector(model=model, conf=conf, device=device)


def cache_path(
    src: Path,
    stride: int,
    conf: float,
    model: str = DEFAULT_MODEL,
    tracker: str = TRACKER,
) -> Path:
    name = f"{src.stem}__s{stride}_c{conf:g}_{Path(model).stem}_{Path(tracker).stem}"
    return TRACK_DIR / f"{name}.parquet"


def normalize(dets: pd.DataFrame) -> pd.DataFrame:
    out = dets.reindex(columns=COLUMNS)
    return out.astype(DTYPES)


def tracks(
    src: Path,
    stride: int = TRACK_STRIDE,
    conf: float = 0.25,
    model: str = DEFAULT_MODEL,
    tracker: str = TRACKER,
    force: bool = False,
) -> pd.DataFrame:
    """Tracks for one clip, cached. An empty result is cached too."""
    path = cache_path(src, stride, conf, model, tracker)
    if path.exists() and not force:
        return normalize(pd.read_parquet(path))

    dets = get_detector(model=model, conf=conf).track(
        src, stride=stride, tracker=tracker
    )
    dets = normalize(dets)

    atomic_write(path, lambda out: dets.to_parquet(out, index=False))
    return dets


FACE_WEIGHTS = MODEL_DIR / "face_detection_yunet_2023mar.onnx"
FACE_WEIGHTS_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)

FACE_COLUMNS = ["frame", "conf", "x1", "y1", "x2", "y2", "cx", "cy"]


def ensure_face_weights() -> Path:
    """YuNet ONNX, downloaded into cache/ on first use."""
    if FACE_WEIGHTS.exists() and FACE_WEIGHTS.stat().st_size > 0:
        return FACE_WEIGHTS

    import urllib.request

    def write(out: Path) -> None:
        urllib.request.urlretrieve(FACE_WEIGHTS_URL, out)

    return atomic_write(FACE_WEIGHTS, write)


def face_cache_path(src: Path, stride: int, width: int, min_score: float) -> Path:
    return FACE_DIR / f"{src.stem}__s{stride}_w{width}_c{min_score:g}.parquet"


def detect_faces(src: Path, stride: int, width: int, min_score: float) -> pd.DataFrame:
    """Faces on every stride-th frame, in source pixels. Frames are shrunk to `width`."""
    import cv2

    ensure_face_weights()
    cap = cv2.VideoCapture(str(src))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = width / frame_w
    size = (width, round(frame_h * scale))
    detector = cv2.FaceDetectorYN.create(str(FACE_WEIGHTS), "", size, min_score)

    rows, frame = [], 0
    try:
        # grab() skips frames cheaply while keeping the tracker's frame numbers.
        # Same frames Ultralytics decodes (stride-1, 2*stride-1, ...), so faces
        # and bodies can be joined on `frame`.
        while cap.grab():
            if (frame + 1) % stride == 0:
                ok, image = cap.retrieve()
                if not ok:
                    break
                _, found = detector.detect(cv2.resize(image, size))
                for f in found if found is not None else []:
                    x, y, w, h = (f[:4] / scale).tolist()
                    rows.append(
                        (frame, float(f[-1]), x, y, x + w, y + h, x + w / 2, y + h / 2)
                    )
            frame += 1
    finally:
        cap.release()
    return pd.DataFrame(rows, columns=FACE_COLUMNS)


def faces(
    src: Path, stride: int, width: int, min_score: float, force: bool = False
) -> pd.DataFrame:
    """Faces for one clip, cached."""
    path = face_cache_path(src, stride, width, min_score)
    if path.exists() and not force:
        return pd.read_parquet(path)

    found = detect_faces(src, stride, width, min_score)
    atomic_write(path, lambda out: found.to_parquet(out, index=False))
    return found
