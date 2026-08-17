#!/usr/bin/env python3
import argparse
import csv
import shutil
from pathlib import Path


DEFAULT_FILES_TO_COPY = ("mixture.wav", "speech_gt.wav", "concert_gt.wav")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Copy sampleN folders from a generate_test_wavs_speech_concert_bird.py "
            "2mix output directory by looking up labels in the source 2mix CSV."
        )
    )
    parser.add_argument(
        "--csv_path",
        type=Path,
        required=True,
        help="2mix CSV used to generate the sampleN wav directory",
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Directory containing sampleN folders",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Destination directory for the filtered sampleN folders",
    )
    parser.add_argument(
        "--target_labels",
        nargs=2,
        default=("speech", "concert"),
        metavar=("LABEL1", "LABEL2"),
        help="Two labels to select, order-insensitive. Default: speech concert",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite destination sample folders if they already exist",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print selected samples without copying",
    )
    parser.add_argument(
        "--manifest_name",
        default="selected_samples.csv",
        help="Manifest filename written under output_dir. Default: selected_samples.csv",
    )
    parser.add_argument(
        "--copy_all_files",
        action="store_true",
        help="Copy the whole sampleN folder instead of only mixture.wav, speech_gt.wav, and concert_gt.wav",
    )
    return parser.parse_args()


def normalize_label(label):
    return str(label).strip().lower()


def row_labels(row):
    try:
        return {
            normalize_label(row["s1_label"]),
            normalize_label(row["s2_label"]),
        }
    except KeyError as exc:
        raise KeyError(
            "CSV must contain s1_label and s2_label columns for 2mix filtering"
        ) from exc


def copy_sample_dir(
    src_dir,
    dst_dir,
    files_to_copy=DEFAULT_FILES_TO_COPY,
    overwrite=False,
    dry_run=False,
    copy_all_files=False,
):
    if not src_dir.is_dir():
        return "missing"

    if dry_run:
        return "planned"

    if copy_all_files:
        if dst_dir.exists():
            if not overwrite:
                return "exists"
            shutil.rmtree(dst_dir)

        shutil.copytree(src_dir, dst_dir)
        return "copied"

    if dst_dir.exists():
        if not overwrite:
            return "exists"

    dst_dir.mkdir(parents=True, exist_ok=True)
    for filename in files_to_copy:
        src_file = src_dir / filename
        if not src_file.is_file():
            return "missing"
        shutil.copy2(src_file, dst_dir / filename)

    return "copied"


def write_manifest(manifest_path, selected_rows):
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sample_index", "sample_name", "s1_label", "s2_label"],
        )
        writer.writeheader()
        writer.writerows(selected_rows)


def main():
    args = parse_args()

    if not args.csv_path.is_file():
        raise FileNotFoundError(f"csv_path does not exist: {args.csv_path}")
    if not args.input_dir.is_dir():
        raise FileNotFoundError(f"input_dir does not exist: {args.input_dir}")

    target_labels = {normalize_label(label) for label in args.target_labels}
    if len(target_labels) != 2:
        raise ValueError("--target_labels must contain two different labels")

    counts = {
        "csv_rows": 0,
        "matched": 0,
        "copied": 0,
        "planned": 0,
        "exists": 0,
        "missing": 0,
    }
    selected_rows = []

    with args.csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        for index, row in enumerate(reader):
            counts["csv_rows"] += 1
            labels = row_labels(row)
            if labels != target_labels:
                continue

            counts["matched"] += 1
            sample_name = f"sample{index}"
            src_dir = args.input_dir / sample_name
            dst_dir = args.output_dir / sample_name
            result = copy_sample_dir(
                src_dir,
                dst_dir,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
                copy_all_files=args.copy_all_files,
            )
            counts[result] += 1

            selected_rows.append(
                {
                    "sample_index": index,
                    "sample_name": sample_name,
                    "s1_label": normalize_label(row["s1_label"]),
                    "s2_label": normalize_label(row["s2_label"]),
                }
            )

            if args.dry_run:
                print(f"[plan] {src_dir} -> {dst_dir}")
            elif result in ("exists", "missing"):
                print(f"[{result}] {src_dir} -> {dst_dir}")

    if not args.dry_run:
        write_manifest(args.output_dir / args.manifest_name, selected_rows)

    print("\nFilter summary")
    print(f"CSV rows scanned: {counts['csv_rows']}")
    print(f"Samples matched:  {counts['matched']}")
    print(f"Samples copied:   {counts['copied']}")
    print(f"Samples planned:  {counts['planned']}")
    print(f"Samples existing: {counts['exists']}")
    print(f"Samples missing:  {counts['missing']}")


if __name__ == "__main__":
    main()
