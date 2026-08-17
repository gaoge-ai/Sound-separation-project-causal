import argparse
import csv
import random
from pathlib import Path

SOURCE_CATEGORIES = ("speech", "concert")
REQUIRED_COLUMNS = ("filename", "label", "category")


def load_and_validate_csv(csv_path, expected_category):
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
            raise ValueError(f"{csv_path}:{row_idx} 的 category 非法: {category}")
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


def build_output_columns():
    column_names = []
    for i in range(1, 3):
        column_names.extend([f"s{i}_path", f"s{i}_label", f"s{i}_snr"])
    return column_names


def generate_2mix_csv(output_csv, speech_rows, concert_rows, num_samples, rng):
    output_rows = []

    for _ in range(num_samples):
        selected_samples = [
            rng.choice(speech_rows),
            rng.choice(concert_rows),
        ]
        rng.shuffle(selected_samples)

        row_data = []
        for sample in selected_samples:
            snr_db = rng.uniform(-3.0, 3.0)
            row_data.extend([sample["audio_path"], sample["category"], snr_db])
        output_rows.append(row_data)

    output_columns = build_output_columns()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(output_columns)
        writer.writerows(output_rows)

    print(f"已生成 {num_samples} 条 speech+concert 2mix 数据到 {output_csv}")
    print(f"数据形状: ({len(output_rows)}, {len(output_columns)})")
    print("生成模式: 每条样本包含 speech 和 concert 各一个源，源顺序随机打乱")


def main():
    parser = argparse.ArgumentParser(description="生成 speech+concert 2mix CSV 文件")
    parser.add_argument("--speech_csv", type=str, help="speech 输入 CSV 路径")
    parser.add_argument("--concert_csv", type=str, help="concert 输入 CSV 路径")
    parser.add_argument("--output_csv", type=str, help="输出 CSV 文件路径")
    parser.add_argument("--num_samples", type=int, help="输出样本数")
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="随机种子",
    )

    args = parser.parse_args()

    missing_args = [
        arg_name
        for arg_name, value in (
            ("--speech_csv", args.speech_csv),
            ("--concert_csv", args.concert_csv),
            ("--output_csv", args.output_csv),
            ("--num_samples", args.num_samples),
        )
        if value is None
    ]
    if missing_args:
        parser.error("生成 2mix CSV 时需要参数: " + ", ".join(missing_args))

    if args.num_samples <= 0:
        parser.error("--num_samples 必须为正整数")

    speech_csv = Path(args.speech_csv)
    concert_csv = Path(args.concert_csv)
    output_csv = Path(args.output_csv)
    rng = random.Random(args.random_seed)

    print(f"开始生成 speech+concert 2mix 数据，样本数: {args.num_samples}")
    print(f"随机 seed: {args.random_seed}")

    speech_rows = load_and_validate_csv(speech_csv, "speech")
    concert_rows = load_and_validate_csv(concert_csv, "concert")
    generate_2mix_csv(
        output_csv=output_csv,
        speech_rows=speech_rows,
        concert_rows=concert_rows,
        num_samples=args.num_samples,
        rng=rng,
    )


if __name__ == "__main__":
    main()
