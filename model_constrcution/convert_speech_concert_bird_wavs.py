#!/usr/bin/env python3
import argparse
import re
import shutil
from pathlib import Path


SAMPLE_DIR_RE = re.compile(r"^sample(\d+)$")

FILE_MAPPINGS = [
    ("mixture.wav", "mixture"),
    ("speech_gt.wav", "speech"),
    ("concert_gt.wav", "music"),
    ("bird_gt.wav", "noise"),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Copy speech/concert/bird sampleN wav folders into a "
            "dataset_root/{split}/{mixture,speech,music,noise}/... layout."
        )
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Directory containing sampleN folders",
    )
    parser.add_argument(
        "--dataset_root",
        type=Path,
        required=True,
        help="Destination dataset root",
    )
    parser.add_argument(
        "--split",
        choices=("train", "valid", "test"),
        default="test",
        help="Destination split under dataset_root. Default: test",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite destination wav files if they already exist",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print planned copies without creating directories or copying files",
    )
    return parser.parse_args()


def iter_sample_dirs(input_dir):
    for path in input_dir.iterdir():
        if not path.is_dir():
            continue

        match = SAMPLE_DIR_RE.match(path.name)
        if match is None:
            continue

        yield int(match.group(1)), path


def destination_path(dataset_root, split, category, index):
    return (
        dataset_root
        / split
        / category
        / f"{category}{index}"
        / f"{category}_{index}.wav"
    )


def copy_one(src_path, dst_path, overwrite=False, dry_run=False):
    if not src_path.is_file():
        return "missing"

    if dst_path.exists() and not overwrite:
        return "exists"

    if dry_run:
        return "planned"

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dst_path)
    return "copied"


def main():
    args = parse_args()

    if not args.input_dir.is_dir():
        raise FileNotFoundError(f"input_dir does not exist: {args.input_dir}")

    sample_dirs = sorted(iter_sample_dirs(args.input_dir), key=lambda item: item[0])
    if not sample_dirs:
        raise ValueError(f"No sampleN directories found under: {args.input_dir}")

    counts = {
        "samples": 0,
        "copied": 0,
        "planned": 0,
        "exists": 0,
        "missing": 0,
    }

    for index, sample_dir in sample_dirs:
        counts["samples"] += 1

        for src_name, dst_category in FILE_MAPPINGS:
            src_path = sample_dir / src_name
            dst_path = destination_path(
                args.dataset_root,
                args.split,
                dst_category,
                index,
            )
            result = copy_one(
                src_path,
                dst_path,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
            )
            counts[result] += 1

            if result in ("missing", "exists"):
                print(f"[{result}] {src_path} -> {dst_path}")
            elif args.dry_run:
                print(f"[plan] {src_path} -> {dst_path}")

    print("\nConversion summary")
    print(f"Samples scanned: {counts['samples']}")
    print(f"Files copied:    {counts['copied']}")
    print(f"Files planned:   {counts['planned']}")
    print(f"Files existing:  {counts['exists']}")
    print(f"Files missing:   {counts['missing']}")


if __name__ == "__main__":
    main()
