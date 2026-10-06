"""
End-to-end pipeline: probe -> plan (runs detection) -> render.

    from smartcrop.pipeline import reframe
    reframe("input.mp4", "output.mp4")
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from smartcrop.plan import DEFAULT_PLANNER, PLANNERS, Decision, Planner
from smartcrop.probe import ensure_upright, probe
from smartcrop.render import render
from smartcrop.utils import list_videos, portrait_name


def reframe(
    src: Path,
    dst: Path,
    planner: Planner = PLANNERS[DEFAULT_PLANNER],
    meta: dict | None = None,
) -> Decision:
    """One clip: probe it, plan the crop path, render it."""
    src, dst = Path(src), Path(dst)
    if not src.exists():
        raise FileNotFoundError(f"input video not found: {src}")
    src, meta = ensure_upright(src)
    decision = planner(src, meta)
    render(src, dst, decision.x, meta)
    return decision


def reframe_all(
    videos: Path, out: Path, planner_name: str = DEFAULT_PLANNER, force: bool = False
) -> pd.DataFrame:
    """
    Every video in `videos` into `out/<planner>/`, skipping ones already rendered.

    Each clip's decision is written to `decisions.csv` beside the outputs.
    """
    from tqdm import tqdm

    planner = PLANNERS[planner_name]
    target = Path(out) / planner_name
    rows = []
    for src in tqdm(list_videos(Path(videos)), desc=planner_name):
        meta = probe(src)
        dst = target / portrait_name(src)
        if force or not dst.exists():
            decision = reframe(src, dst, planner, meta)
        else:
            decision = planner(src, meta)
        rows.append(
            {
                "file": src.name,
                "output": dst.name,
                "mode": decision.mode,
                "span": round(decision.span, 4),
                "reason": decision.reason,
            }
        )

    decisions = pd.DataFrame(rows)
    if rows:
        target.mkdir(parents=True, exist_ok=True)
        decisions.to_csv(target / "decisions.csv", index=False)
    return decisions
