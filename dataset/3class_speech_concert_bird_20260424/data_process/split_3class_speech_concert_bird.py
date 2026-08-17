import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


SPLITS = ("train", "valid", "test")
DEFAULT_SPLIT_RATIOS = {"train": 0.90, "valid": 0.05, "test": 0.05}
SEGMENT_SUFFIX_PATTERN = re.compile(r"_seg\d.*$")


SOURCE_SPECS = (
    {
        "source_name": "XC_Clean_199h_cut",
        "relative_path": "XC_Clean_199h_cut.csv",
        "class_name": "bird",
        "format": "csv",
        "group_rule": "original_utt",
    },
    {
        "source_name": "bird_vgg",
        "relative_path": "bird_vgg.csv",
        "class_name": "bird",
        "format": "csv",
        "group_rule": "utt",
    },
    {
        "source_name": "concert_song_cut",
        "relative_path": "concert_song_cut.csv",
        "class_name": "concert",
        "format": "csv",
        "group_rule": "original_utt",
    },
    {
        "source_name": "song_cut_202604221637",
        "relative_path": "song_cut_202604221637.csv",
        "class_name": "concert",
        "format": "csv",
        "group_rule": "original_utt",
    },
    {
        "source_name": "aishell2_segments",
        "relative_path": "manifests/aishell2_segments.jsonl",
        "class_name": "speech",
        "format": "jsonl",
        "group_rule": "speaker",
    },
    {
        "source_name": "aishell_segments",
        "relative_path": "manifests/aishell_segments.jsonl",
        "class_name": "speech",
        "format": "jsonl",
        "group_rule": "speaker",
    },
    {
        "source_name": "kespeech_segments",
        "relative_path": "manifests/kespeech_segments.jsonl",
        "class_name": "speech",
        "format": "jsonl",
        "group_rule": "speaker",
    },
    {
        "source_name": "magic_data_segments",
        "relative_path": "manifests/magic_data_segments.jsonl",
        "class_name": "speech",
        "format": "jsonl",
        "group_rule": "speaker",
    },
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="按 group-level 规则切分 speech/concert/bird 单类数据"
    )
    parser.add_argument(
        "--input_root",
        type=str,
        default="dataset/3class_speech_concert_bird_20260424",
        help="原始单类数据根目录",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="输出目录；默认放到 input_root/splits_seed<seed>_90505",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="固定随机种子，用于可复现划分",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=DEFAULT_SPLIT_RATIOS["train"],
        help="train group 占比",
    )
    parser.add_argument(
        "--valid_ratio",
        type=float,
        default=DEFAULT_SPLIT_RATIOS["valid"],
        help="valid group 占比",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=DEFAULT_SPLIT_RATIOS["test"],
        help="test group 占比",
    )
    return parser.parse_args()


def validate_ratios(train_ratio, valid_ratio, test_ratio):
    ratio_sum = train_ratio + valid_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-8:
        raise ValueError(
            f"train/valid/test 比例之和必须为 1.0，当前为 {ratio_sum:.8f}"
        )
    for split_name, ratio in (
        ("train", train_ratio),
        ("valid", valid_ratio),
        ("test", test_ratio),
    ):
        if ratio <= 0:
            raise ValueError(f"{split_name}_ratio 必须大于 0，当前为 {ratio}")


def derive_class_seed(base_seed, class_name):
    digest = hashlib.sha256(f"{base_seed}:{class_name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def format_ratio_tag(ratio):
    return str(int(round(ratio * 100)))


def extract_original_utt_id(audio_path):
    stem = Path(audio_path).stem
    return SEGMENT_SUFFIX_PATTERN.sub("", stem)


def build_group_id(source_name, group_rule, row):
    audio_path = row["filename"]

    if group_rule == "speaker":
        speaker_id = str(row["speaker_id"]).strip()
        if not speaker_id:
            raise ValueError(f"{source_name} 中存在空 speaker_id: {audio_path}")
        dataset_name = str(row.get("dataset", source_name)).strip()
        return f"{source_name}:{dataset_name}:{speaker_id}"

    if group_rule == "original_utt":
        original_utt = extract_original_utt_id(audio_path)
        if not original_utt:
            raise ValueError(f"{source_name} 无法从路径解析原始 utt: {audio_path}")
        return f"{source_name}:{original_utt}"

    if group_rule == "utt":
        return f"{source_name}:{Path(audio_path).stem}"

    raise ValueError(f"未知 group_rule: {group_rule}")


def load_csv_rows(csv_path, source_name, class_name, group_rule):
    rows = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for line_number, fields in enumerate(reader, start=1):
            if not fields:
                continue
            if len(fields) != 3:
                raise ValueError(
                    f"{csv_path}:{line_number} 期望 3 列，实际为 {len(fields)}"
                )
            filename, label, category = [str(item).strip() for item in fields]
            if category != class_name:
                raise ValueError(
                    f"{csv_path}:{line_number} 的 category={category}，应为 {class_name}"
                )
            row = {
                "filename": filename,
                "label": label,
                "category": category,
                "source": source_name,
            }
            row["group_id"] = build_group_id(source_name, group_rule, row)
            rows.append(row)
    return rows


def load_jsonl_rows(jsonl_path, source_name, class_name, group_rule):
    rows = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            row = {
                "filename": str(payload["wav_path"]).strip(),
                "label": class_name,
                "category": class_name,
                "source": source_name,
                "speaker_id": str(payload["speaker_id"]).strip(),
                "dataset": str(payload.get("dataset", source_name)).strip(),
            }
            row["group_id"] = build_group_id(source_name, group_rule, row)
            rows.append(row)
    return rows


def load_all_rows(input_root):
    rows_by_class = defaultdict(list)
    for spec in SOURCE_SPECS:
        data_path = input_root / spec["relative_path"]
        if not data_path.exists():
            raise FileNotFoundError(f"缺少输入文件: {data_path}")

        if spec["format"] == "csv":
            source_rows = load_csv_rows(
                csv_path=data_path,
                source_name=spec["source_name"],
                class_name=spec["class_name"],
                group_rule=spec["group_rule"],
            )
        elif spec["format"] == "jsonl":
            source_rows = load_jsonl_rows(
                jsonl_path=data_path,
                source_name=spec["source_name"],
                class_name=spec["class_name"],
                group_rule=spec["group_rule"],
            )
        else:
            raise ValueError(f"未知文件格式: {spec['format']}")

        rows_by_class[spec["class_name"]].extend(source_rows)
    return rows_by_class


def assign_groups_to_splits(group_ids, rng, ratios):
    sorted_group_ids = sorted(group_ids)
    shuffled_group_ids = list(sorted_group_ids)
    rng.shuffle(shuffled_group_ids)

    total_groups = len(shuffled_group_ids)
    valid_count = int(total_groups * ratios["valid"])
    test_count = int(total_groups * ratios["test"])
    train_count = total_groups - valid_count - test_count

    split_to_groups = {
        "train": shuffled_group_ids[:train_count],
        "valid": shuffled_group_ids[train_count : train_count + valid_count],
        "test": shuffled_group_ids[train_count + valid_count :],
    }

    group_to_split = {}
    for split_name, split_group_ids in split_to_groups.items():
        for group_id in split_group_ids:
            group_to_split[group_id] = split_name
    return group_to_split


def split_rows_by_class(rows_by_class, seed, ratios):
    split_rows = {split_name: defaultdict(list) for split_name in SPLITS}
    summary = {"seed": seed, "ratios": ratios, "classes": {}}

    for class_name, class_rows in sorted(rows_by_class.items()):
        class_rng = random.Random(derive_class_seed(seed, class_name))
        group_ids = {row["group_id"] for row in class_rows}
        group_to_split = assign_groups_to_splits(
            group_ids=group_ids,
            rng=class_rng,
            ratios=ratios,
        )

        per_split_group_counter = {split_name: set() for split_name in SPLITS}
        per_split_source_counter = {split_name: Counter() for split_name in SPLITS}

        for row in class_rows:
            split_name = group_to_split[row["group_id"]]
            split_rows[split_name][class_name].append(row)
            per_split_group_counter[split_name].add(row["group_id"])
            per_split_source_counter[split_name][row["source"]] += 1

        class_summary = {
            "num_rows": len(class_rows),
            "num_groups": len(group_ids),
            "splits": {},
        }
        for split_name in SPLITS:
            split_class_rows = split_rows[split_name][class_name]
            class_summary["splits"][split_name] = {
                "num_rows": len(split_class_rows),
                "num_groups": len(per_split_group_counter[split_name]),
                "source_row_counts": dict(sorted(per_split_source_counter[split_name].items())),
            }
        summary["classes"][class_name] = class_summary

    return split_rows, summary


def write_split_csvs(output_dir, split_rows):
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = ["filename", "label", "category", "source", "group_id"]

    for split_name in SPLITS:
        for class_name in ("speech", "concert", "bird"):
            output_csv = output_dir / f"{class_name}_{split_name}.csv"
            rows = split_rows[split_name][class_name]
            with output_csv.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for row in sorted(rows, key=lambda item: (item["source"], item["filename"])):
                    writer.writerow({key: row[key] for key in fieldnames})


def write_summary(output_dir, summary):
    summary_path = output_dir / "split_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


def print_summary(summary, output_dir):
    print(f"划分完成，输出目录: {output_dir}")
    print(f"seed: {summary['seed']}")
    print(
        "ratios: "
        f"train={summary['ratios']['train']:.4f}, "
        f"valid={summary['ratios']['valid']:.4f}, "
        f"test={summary['ratios']['test']:.4f}"
    )
    for class_name in ("speech", "concert", "bird"):
        class_summary = summary["classes"][class_name]
        print(
            f"[{class_name}] rows={class_summary['num_rows']}, "
            f"groups={class_summary['num_groups']}"
        )
        for split_name in SPLITS:
            split_summary = class_summary["splits"][split_name]
            print(
                f"  {split_name}: rows={split_summary['num_rows']}, "
                f"groups={split_summary['num_groups']}"
            )


def main():
    args = parse_args()
    validate_ratios(args.train_ratio, args.valid_ratio, args.test_ratio)

    input_root = Path(args.input_root)
    if not input_root.exists():
        raise FileNotFoundError(f"--input_root 不存在: {input_root}")

    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else input_root
        / (
            f"splits_seed{args.seed}_"
            f"{format_ratio_tag(args.train_ratio)}"
            f"{format_ratio_tag(args.valid_ratio)}"
            f"{format_ratio_tag(args.test_ratio)}"
        )
    )
    ratios = {
        "train": args.train_ratio,
        "valid": args.valid_ratio,
        "test": args.test_ratio,
    }

    rows_by_class = load_all_rows(input_root=input_root)
    split_rows, summary = split_rows_by_class(
        rows_by_class=rows_by_class,
        seed=args.seed,
        ratios=ratios,
    )
    write_split_csvs(output_dir=output_dir, split_rows=split_rows)
    write_summary(output_dir=output_dir, summary=summary)
    print_summary(summary=summary, output_dir=output_dir)


if __name__ == "__main__":
    main()
