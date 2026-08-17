#!/usr/bin/env python3
"""Create an epoch-axis TensorBoard view from existing event logs.

The original TensorBoard event files are left untouched. This script reads scalar
events from an existing logdir and writes a new logdir whose scalar step is the
epoch index, which makes epoch-level metrics easier to compare across runs with
different batch sizes.

Typical usage:
    python model_constrcution/rewrite_tb_epoch_axis.py \
        --input model_constrcution/experiments \
        --output model_constrcution/tb_view_epoch

Then open:
    tensorboard --logdir model_constrcution/tb_view_epoch
"""

from __future__ import annotations

import argparse
import fnmatch
import math
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:
    from tensorboard.backend.event_processing import event_accumulator
    from tensorboard.compat.proto.event_pb2 import Event
    from tensorboard.compat.proto.summary_pb2 import Summary
    from tensorboard.summary.writer.event_file_writer import EventFileWriter
except ModuleNotFoundError as exc:  # pragma: no cover - exercised by user env.
    raise SystemExit(
        "This script requires the 'tensorboard' Python package. Run it in the "
        "same environment you use for `tensorboard --logdir ...`, or install "
        "it with `pip install tensorboard`."
    ) from exc


EVENT_FILE_PREFIX = "events.out.tfevents"
DEFAULT_INCLUDE_PATTERNS = ("val_*", "*_epoch")
DEFAULT_EXCLUDE_PATTERNS = ("epoch", "hp_metric")


ScalarPoint = Tuple[float, int, float]


def find_event_dirs(root: Path) -> List[Path]:
    """Return directories that directly contain TensorBoard event files."""
    event_dirs: List[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_dir():
            continue
        try:
            has_event_file = any(
                child.is_file() and child.name.startswith(EVENT_FILE_PREFIX)
                for child in path.iterdir()
            )
        except PermissionError:
            continue
        if has_event_file:
            event_dirs.append(path)

    if any(
        child.is_file() and child.name.startswith(EVENT_FILE_PREFIX)
        for child in root.iterdir()
    ):
        event_dirs.insert(0, root)

    # Preserve order while removing duplicates when root itself was already
    # returned by rglob in unusual pathlib implementations.
    seen = set()
    unique_event_dirs: List[Path] = []
    for event_dir in event_dirs:
        resolved = event_dir.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_event_dirs.append(event_dir)
    return unique_event_dirs


def load_scalar_tags(event_dir: Path) -> Mapping[str, List[ScalarPoint]]:
    """Load all scalar time series from one TensorBoard run directory."""
    accumulator = event_accumulator.EventAccumulator(
        str(event_dir),
        size_guidance={event_accumulator.SCALARS: 0},
    )
    accumulator.Reload()

    scalars: Dict[str, List[ScalarPoint]] = {}
    for tag in accumulator.Tags().get("scalars", []):
        scalars[tag] = [
            (float(point.wall_time), int(point.step), float(point.value))
            for point in accumulator.Scalars(tag)
        ]
    return scalars


def matches_any(tag: str, patterns: Sequence[str]) -> bool:
    return any(fnmatch.fnmatchcase(tag, pattern) for pattern in patterns)


def select_tags(
    tags: Iterable[str],
    include_patterns: Sequence[str],
    exclude_patterns: Sequence[str],
) -> List[str]:
    selected = []
    for tag in sorted(tags):
        if matches_any(tag, exclude_patterns):
            continue
        if matches_any(tag, include_patterns):
            selected.append(tag)
    return selected


def clean_epoch_value(value: float) -> int:
    """Convert a logged epoch scalar to a TensorBoard integer step."""
    if not math.isfinite(value):
        raise ValueError(f"Epoch value is not finite: {value}")
    return int(round(value))


def build_epoch_step_map(epoch_points: Sequence[ScalarPoint]) -> Dict[int, int]:
    """Build a mapping from original global_step to logged epoch value."""
    epoch_by_step: Dict[int, int] = {}
    for _, step, value in epoch_points:
        epoch_by_step[step] = clean_epoch_value(value)
    return epoch_by_step


def infer_epoch_for_point(
    original_step: int,
    point_index: int,
    epoch_by_step: Mapping[int, int],
    epoch_points: Sequence[ScalarPoint],
    fallback: str,
) -> int:
    """Map one scalar point from original global_step to epoch index."""
    if original_step in epoch_by_step:
        return epoch_by_step[original_step]

    if fallback == "index":
        return point_index

    if fallback == "nearest":
        if not epoch_points:
            return point_index
        _, _, nearest_epoch = min(
            epoch_points,
            key=lambda point: (abs(point[1] - original_step), point[1]),
        )
        return clean_epoch_value(nearest_epoch)

    raise ValueError(f"Unsupported fallback: {fallback}")


def write_scalar_series(
    writer: EventFileWriter,
    tag: str,
    points: Sequence[Tuple[float, int, float]],
) -> None:
    for wall_time, step, value in points:
        event = Event(
            wall_time=wall_time,
            step=int(step),
            summary=Summary(
                value=[Summary.Value(tag=tag, simple_value=float(value))]
            ),
        )
        writer.add_event(event)


def convert_run(
    event_dir: Path,
    output_dir: Path,
    include_patterns: Sequence[str],
    exclude_patterns: Sequence[str],
    fallback: str,
    epoch_offset: int,
    keep_tag_names: bool,
) -> Tuple[int, int, bool]:
    """Convert one run directory.

    Returns:
        (number_of_selected_tags, number_of_written_points, used_epoch_tag)
    """
    scalars = load_scalar_tags(event_dir)
    if not scalars:
        return 0, 0, False

    selected_tags = select_tags(scalars.keys(), include_patterns, exclude_patterns)
    if not selected_tags:
        return 0, 0, "epoch" in scalars

    epoch_points = scalars.get("epoch", [])
    epoch_by_step = build_epoch_step_map(epoch_points)
    used_epoch_tag = bool(epoch_points)

    output_dir.mkdir(parents=True, exist_ok=True)
    writer = EventFileWriter(str(output_dir))

    written_points = 0
    try:
        for tag in selected_tags:
            converted_points: List[ScalarPoint] = []
            for point_index, (wall_time, original_step, value) in enumerate(scalars[tag]):
                epoch_step = infer_epoch_for_point(
                    original_step=original_step,
                    point_index=point_index,
                    epoch_by_step=epoch_by_step,
                    epoch_points=epoch_points,
                    fallback=fallback,
                )
                converted_points.append((wall_time, epoch_step + epoch_offset, value))

            output_tag = tag if keep_tag_names else f"{tag}_by_epoch"
            write_scalar_series(writer, output_tag, converted_points)
            written_points += len(converted_points)
    finally:
        writer.flush()
        writer.close()

    return len(selected_tags), written_points, used_epoch_tag


def parse_patterns(raw_patterns: Optional[Sequence[str]], defaults: Sequence[str]) -> List[str]:
    if raw_patterns is None:
        return list(defaults)
    patterns: List[str] = []
    for raw_pattern in raw_patterns:
        patterns.extend(part.strip() for part in raw_pattern.split(",") if part.strip())
    return patterns


def make_output_run_dir(input_root: Path, event_dir: Path, output_root: Path) -> Path:
    try:
        relative = event_dir.relative_to(input_root)
    except ValueError:
        relative = Path(event_dir.name)
    if str(relative) == ".":
        return output_root
    return output_root / relative


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Rewrite existing TensorBoard scalar logs into a new logdir where "
            "selected epoch-level metrics use epoch as the TensorBoard step."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Original TensorBoard logdir or a parent directory containing runs.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="New display logdir to write. Original logs are not modified.",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=None,
        help=(
            "Scalar tag glob(s) to convert. Can be repeated or comma-separated. "
            "Default: val_*,*_epoch"
        ),
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=None,
        help=(
            "Scalar tag glob(s) to skip. Can be repeated or comma-separated. "
            "Default: epoch,hp_metric"
        ),
    )
    parser.add_argument(
        "--fallback",
        choices=("index", "nearest"),
        default="index",
        help=(
            "How to assign epochs when a metric point has no exact matching "
            "'epoch' scalar at the same original step. 'index' is safest for "
            "epoch-level metrics; 'nearest' uses the closest logged epoch step."
        ),
    )
    parser.add_argument(
        "--epoch-offset",
        type=int,
        default=0,
        help="Add this integer offset to output epoch steps. Use 1 for one-based epochs.",
    )
    parser.add_argument(
        "--suffix-tags",
        action="store_true",
        help="Write tags as '<original_tag>_by_epoch' instead of keeping original names.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the output logdir before writing.",
    )

    args = parser.parse_args(argv)

    input_root = args.input.expanduser().resolve()
    output_root = args.output.expanduser().resolve()

    if not input_root.exists():
        raise SystemExit(f"Input logdir does not exist: {input_root}")
    if input_root == output_root:
        raise SystemExit("--output must be different from --input")
    if output_root.exists():
        if args.overwrite:
            shutil.rmtree(output_root)
        elif any(output_root.iterdir()):
            raise SystemExit(
                f"Output logdir already exists and is not empty: {output_root}\n"
                "Use --overwrite to refresh it, or choose a new --output path."
            )
    output_root.mkdir(parents=True, exist_ok=True)

    include_patterns = parse_patterns(args.include, DEFAULT_INCLUDE_PATTERNS)
    exclude_patterns = parse_patterns(args.exclude, DEFAULT_EXCLUDE_PATTERNS)

    event_dirs = find_event_dirs(input_root)
    if not event_dirs:
        raise SystemExit(f"No TensorBoard event files found under: {input_root}")

    total_runs = 0
    total_tags = 0
    total_points = 0
    runs_without_epoch_tag = []

    for event_dir in event_dirs:
        output_dir = make_output_run_dir(input_root, event_dir, output_root)
        tag_count, point_count, used_epoch_tag = convert_run(
            event_dir=event_dir,
            output_dir=output_dir,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
            fallback=args.fallback,
            epoch_offset=args.epoch_offset,
            keep_tag_names=not args.suffix_tags,
        )
        if tag_count == 0:
            continue
        total_runs += 1
        total_tags += tag_count
        total_points += point_count
        if not used_epoch_tag:
            runs_without_epoch_tag.append(event_dir)
        print(
            f"[ok] {event_dir.relative_to(input_root)} -> "
            f"{output_dir.relative_to(output_root)} "
            f"({tag_count} tags, {point_count} points)"
        )

    print(
        f"\nWrote {total_points} scalar points from {total_tags} tags "
        f"across {total_runs} runs to: {output_root}"
    )
    if runs_without_epoch_tag:
        print(
            "\nNote: some runs had no explicit 'epoch' scalar. For those runs, "
            "epoch steps were assigned by point order:"
        )
        for event_dir in runs_without_epoch_tag[:20]:
            print(f"  - {event_dir}")
        if len(runs_without_epoch_tag) > 20:
            print(f"  ... and {len(runs_without_epoch_tag) - 20} more")
    print(f"\nOpen with:\n  tensorboard --logdir {output_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
