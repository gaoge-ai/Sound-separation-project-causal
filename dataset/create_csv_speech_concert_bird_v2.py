import argparse
import csv
import hashlib
import random
from collections import Counter
from pathlib import Path

SOURCE_CATEGORIES = ("speech", "concert", "bird")
DEFAULT_NUM_SAMPLES = {"train": 20000, "valid": 5000, "test": 3000}
REQUIRED_COLUMNS = ("filename", "label", "category")


def derive_split_seed(base_seed, split):
    digest = hashlib.sha256(f"{base_seed}:{split}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def load_and_validate_split_csv(csv_path, expected_category):
    if not csv_path.exists():
        raise FileNotFoundError(f"缺少输入 CSV: {csv_path}")

    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError(f"{csv_path} 缺少表头")

        missing_columns = [
            column for column in REQUIRED_COLUMNS if column not in reader.fieldnames
        ]
        if missing_columns:
            raise ValueError(f"{csv_path} 缺少列: {', '.join(missing_columns)}")

        rows = list(reader)

    if not rows:
        raise ValueError(f"{csv_path} 没有可用样本")

    validated_rows = []
    for row_idx, row in enumerate(rows, start=2):
        filename = str(row["filename"]).strip()
        label = str(row["label"])
        category = str(row["category"]).strip().lower()

        if category not in SOURCE_CATEGORIES:
            raise ValueError(
                f"{csv_path}:{row_idx} 的 category 非法: {category}"
            )
        if category != expected_category:
            raise ValueError(
                f"{csv_path}:{row_idx} 的 category={category}，应为 {expected_category}"
            )

        path_obj = Path(filename)
        if not path_obj.is_absolute():
            raise ValueError(
                f"{csv_path}:{row_idx} 的 filename 不是绝对路径: {filename}"
            )

        validated_rows.append(
            {
                "audio_path": filename,
                "label": label,
                "category": category,
            }
        )

    return validated_rows


def load_category_pools(source_root, split):
    category_pools = {}
    for category in SOURCE_CATEGORIES:
        csv_path = source_root / f"{category}_{split}.csv"
        category_pools[category] = load_and_validate_split_csv(csv_path, category)
    return category_pools


def sample_categories(num_sources, rng):
    if num_sources == 1:
        return [rng.choice(SOURCE_CATEGORIES)]

    while True:
        categories = [rng.choice(SOURCE_CATEGORIES) for _ in range(num_sources)]
        if len(set(categories)) > 1:
            return categories


def sample_categories_one_source_per_category(num_sources, rng):
    if num_sources != 3:
        raise ValueError("--one_source_per_category 只能与 --num_sources 3 一起使用")

    categories = list(SOURCE_CATEGORIES)
    rng.shuffle(categories)
    return categories


def sample_sources_for_categories(selected_categories, category_pools, split, rng):
    category_counts = Counter(selected_categories)
    selected_samples = []

    for category in SOURCE_CATEGORIES:
        requested_count = category_counts.get(category, 0)
        if requested_count == 0:
            continue

        pool = category_pools[category]
        if requested_count > len(pool):
            raise ValueError(
                f"{split} split 下类别 {category} 需要抽取 {requested_count} 个不同样本，"
                f"但只有 {len(pool)} 个可用样本"
            )

    used_indices_by_category = {category: set() for category in SOURCE_CATEGORIES}
    for category in selected_categories:
        used_indices = used_indices_by_category[category]
        remaining_indices = [
            index for index in range(len(category_pools[category])) if index not in used_indices
        ]
        if not remaining_indices:
            raise ValueError(
                f"{split} split 下类别 {category} 无法在单条 mixture 内继续不放回抽样"
            )

        sample_index = rng.choice(remaining_indices)
        used_indices.add(sample_index)
        sample = category_pools[category][sample_index]
        selected_samples.append(
            {
                "audio_path": sample["audio_path"],
                "category": sample["category"],
            }
        )

    return selected_samples


def build_output_columns(num_sources):
    column_names = []
    for i in range(1, num_sources + 1):
        column_names.extend([f"s{i}_path", f"s{i}_label", f"s{i}_snr"])
    return column_names


def generate_mixed_csv(output_csv, category_pools, num_samples, num_sources, split, rng, one_source_per_category=False):
    output_rows = []

    for _ in range(num_samples):
        if one_source_per_category:
            selected_categories = sample_categories_one_source_per_category(num_sources, rng)
        else:
            selected_categories = sample_categories(num_sources, rng)

        selected_samples = sample_sources_for_categories(
            selected_categories=selected_categories,
            category_pools=category_pools,
            split=split,
            rng=rng,
        )

        row_data = []
        for sample in selected_samples:
            snr_db = rng.uniform(-3.0, 3.0)
            row_data.extend([sample["audio_path"], sample["category"], snr_db])
        output_rows.append(row_data)

    output_columns = build_output_columns(num_sources)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(output_columns)
        writer.writerows(output_rows)

    print(f"已生成 {num_samples} 条混合音频数据到 {output_csv}")
    print(f"数据形状: ({len(output_rows)}, {len(output_columns)})")
    if num_sources == 1:
        print("生成模式: 1mix，speech/concert/bird 三类随机近似等比例抽取")
    elif one_source_per_category:
        print("生成模式: speech/concert/bird 每类各一个源")


def main():
    parser = argparse.ArgumentParser(description="生成三类混合音频数据的 CSV 文件（v2）")
    parser.add_argument("--source_root", type=str, help="输入 CSV 根目录")
    parser.add_argument("--output_csv", type=str, help="输出 CSV 文件路径")
    parser.add_argument(
        "--type",
        type=str,
        choices=["train", "valid", "test"],
        help="生成的数据集类型",
    )
    parser.add_argument(
        "--num_sources",
        type=int,
        choices=[1, 2, 3, 4, 5],
        help="混合声源数量: 1, 2, 3, 4 或 5",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="输出样本数；不传时使用 split 默认值",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="基础随机种子；脚本会按 split 派生子 seed",
    )
    parser.add_argument(
        "--one_source_per_category",
        action="store_true",
        help="仅用于 3mix：每条样本固定包含 speech、concert、bird 各一个源",
    )

    args = parser.parse_args()

    missing_args = [
        arg_name
        for arg_name, value in (
            ("--source_root", args.source_root),
            ("--output_csv", args.output_csv),
            ("--type", args.type),
            ("--num_sources", args.num_sources),
        )
        if value is None
    ]
    if missing_args:
        parser.error("生成混合 CSV 时需要参数: " + ", ".join(missing_args))

    if args.one_source_per_category and args.num_sources != 3:
        parser.error("--one_source_per_category 只能与 --num_sources 3 一起使用")

    if args.num_samples is not None and args.num_samples <= 0:
        parser.error("--num_samples 必须为正整数")

    source_root = Path(args.source_root)
    if not source_root.exists():
        parser.error(f"--source_root 不存在: {source_root}")
    if not source_root.is_dir():
        parser.error(f"--source_root 不是目录: {source_root}")

    output_csv = Path(args.output_csv)
    num_samples = args.num_samples or DEFAULT_NUM_SAMPLES[args.type]
    split_seed = derive_split_seed(args.random_seed, args.type)
    rng = random.Random(split_seed)

    print(f"开始生成 {args.type} 数据，样本数: {num_samples}")
    print(f"基础 seed: {args.random_seed}，{args.type} 子 seed: {split_seed}")

    category_pools = load_category_pools(source_root=source_root, split=args.type)
    generate_mixed_csv(
        output_csv=output_csv,
        category_pools=category_pools,
        num_samples=num_samples,
        num_sources=args.num_sources,
        split=args.type,
        rng=rng,
        one_source_per_category=args.one_source_per_category,
    )


if __name__ == "__main__":
    main()
