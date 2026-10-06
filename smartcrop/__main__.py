"""
Command line.

    uv run python -m smartcrop convert input.mp4
    uv run python -m smartcrop convert input.mp4 output.mp4
    uv run python -m smartcrop batch
    uv run python -m smartcrop batch --videos data/videos --out outputs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from smartcrop.pipeline import reframe, reframe_all
from smartcrop.plan import DEFAULT_PLANNER, PLANNERS
from smartcrop.utils import OUTPUT_DIR, VIDEO_DIR


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="smartcrop",
        description=(
            "Reframe 16:9 landscape video to 9:16 portrait by cropping. "
            "The crop keeps full source height; the planner chooses x(t)."
        ),
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    convert = sub.add_parser("convert", help="reframe one video")
    convert.add_argument("src", type=Path, help="input video")
    convert.add_argument(
        "dst",
        type=Path,
        nargs="?",
        default=None,
        help="output .mp4 (default: <stem>_portrait.mp4 next to the source)",
    )
    convert.add_argument("--planner", choices=list(PLANNERS), default=DEFAULT_PLANNER)

    batch = sub.add_parser("batch", help="reframe every video in a directory")
    batch.add_argument("--planner", choices=list(PLANNERS), default=DEFAULT_PLANNER)
    batch.add_argument(
        "--videos",
        type=Path,
        default=VIDEO_DIR,
        help="input directory (default: data/videos)",
    )
    batch.add_argument(
        "--out",
        type=Path,
        default=OUTPUT_DIR,
        help="output root; files go in <out>/<planner>/ (default: outputs)",
    )
    batch.add_argument("--force", action="store_true", help="re-render existing outputs")

    args = ap.parse_args(argv)

    if args.cmd == "convert":
        if not args.src.exists():
            sys.exit(f"no such file: {args.src}")
        dst = args.dst or args.src.with_name(f"{args.src.stem}_portrait.mp4")
        if dst.suffix.lower() != ".mp4":
            dst = dst.with_suffix(".mp4")
        decision = reframe(args.src, dst, PLANNERS[args.planner])
        print(f"{args.src.name}: {decision.mode}. {decision.reason}")
        print(f"wrote {dst}")
        return

    decisions = reframe_all(args.videos, args.out, args.planner, args.force)
    if decisions.empty:
        sys.exit(
            f"no videos in {args.videos}. "
            f"Drop .mp4 / .mov files there, or pass --videos."
        )
    print(decisions["mode"].value_counts().to_string())
    print(f"-> {args.out / args.planner}")


if __name__ == "__main__":
    main()
