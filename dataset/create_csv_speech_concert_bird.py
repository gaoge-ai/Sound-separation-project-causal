import argparse
import random
from pathlib import Path

import pandas as pd


DATA_ROOT = Path("dataset/3class_speech_concert_bird")
SPEECH_WAV_ROOT = Path("/work107/luoxiaoxue/data/VGGSound/audios_16k")
BIRD_SOURCE_WEIGHTS = {"xc": 0.85, "vgg": 0.15}
BIRD_SPLIT_SIZES = {"valid": 300, "test": 150}
SOURCE_CATEGORIES = ("speech", "concert", "bird")


def load_split_csv(csv_path):
    return pd.read_csv(csv_path)


def load_bird_source_csv(csv_path):
    return pd.read_csv(csv_path, header=None, names=["filename", "label", "category"])


def normalize_audio_path(sample):
    filename = str(sample["filename"])
    category = str(sample["category"]).strip().lower()

    if category == "speech" and filename.lower().endswith(".mp4"):
        wav_filename = Path(filename).name.replace(".mp4", ".wav")
        return str(SPEECH_WAV_ROOT / wav_filename)

    return filename


def build_bird_splits(random_seed=42):
    rng = random.Random(random_seed)
    bird_xc = load_bird_source_csv(DATA_ROOT / "bird_xc" / "all.csv")
    bird_vgg = load_bird_source_csv(DATA_ROOT / "bird_vgg" / "all.csv")

    bird_sources = {
        "xc": bird_xc.sample(frac=1, random_state=random_seed).reset_index(drop=True),
        "vgg": bird_vgg.sample(frac=1, random_state=random_seed + 1).reset_index(drop=True),
    }

    split_pools = {}
    source_offsets = {"xc": 0, "vgg": 0}

    for split in ("valid", "test"):
        split_size = BIRD_SPLIT_SIZES[split]
        xc_count = int(round(split_size * BIRD_SOURCE_WEIGHTS["xc"]))
        vgg_count = split_size - xc_count
        requested = {"xc": xc_count, "vgg": vgg_count}

        split_frames = []
        for source_name in ("xc", "vgg"):
            start = source_offsets[source_name]
            end = start + requested[source_name]
            source_df = bird_sources[source_name]
            if end > len(source_df):
                raise ValueError(f"bird_{source_name} 数据不足，无法构造 {split} split")
            split_frames.append(source_df.iloc[start:end].copy())
            source_offsets[source_name] = end

        split_df = pd.concat(split_frames, ignore_index=True)
        split_df = split_df.sample(frac=1, random_state=rng.randint(0, 10**9)).reset_index(drop=True)
        split_pools[split] = split_df

    train_frames = []
    for source_name, source_df in bird_sources.items():
        train_frames.append(source_df.iloc[source_offsets[source_name]:].copy())
    split_pools["train"] = pd.concat(train_frames, ignore_index=True)
    split_pools["train"] = split_pools["train"].sample(
        frac=1, random_state=rng.randint(0, 10**9)
    ).reset_index(drop=True)

    return split_pools


def load_category_pools(split):
    speech_df = load_split_csv(DATA_ROOT / f"speech_{split}.csv")
    concert_df = load_split_csv(DATA_ROOT / f"concert_{split}.csv")
    bird_splits = build_bird_splits()

    pools = {
        "speech": speech_df.reset_index(drop=True),
        "concert": concert_df.reset_index(drop=True),
        "bird": bird_splits[split].reset_index(drop=True),
    }

    bird_source_pools = {
        "xc": pools["bird"][pools["bird"]["filename"].astype(str).str.contains("/XC_Clean_WAV/")].reset_index(drop=True),
        "vgg": pools["bird"][pools["bird"]["filename"].astype(str).str.contains("/vgg_birds/")].reset_index(drop=True),
    }

    for category, df in pools.items():
        if df.empty:
            raise ValueError(f"{split} split 下类别 {category} 没有可用样本")

    for source_name, df in bird_source_pools.items():
        if df.empty:
            raise ValueError(f"{split} split 下 bird 来源 {source_name} 没有可用样本")

    return pools, bird_source_pools


def sample_one_source(category_pools, bird_source_pools):
    category = random.choice(SOURCE_CATEGORIES)
    return sample_one_source_from_category(category, category_pools, bird_source_pools)


def sample_one_source_from_category(category, category_pools, bird_source_pools):
    if category == "bird":
        bird_source = random.choices(
            population=["xc", "vgg"],
            weights=[BIRD_SOURCE_WEIGHTS["xc"], BIRD_SOURCE_WEIGHTS["vgg"]],
            k=1,
        )[0]
        sample = bird_source_pools[bird_source].sample(n=1, replace=True).iloc[0]
    else:
        sample = category_pools[category].sample(n=1, replace=True).iloc[0]

    return {
        "audio_path": normalize_audio_path(sample),
        "category": category,
    }


def sample_one_source_per_category(category_pools, bird_source_pools):
    selected_samples = [
        sample_one_source_from_category(category, category_pools, bird_source_pools)
        for category in SOURCE_CATEGORIES
    ]
    random.shuffle(selected_samples)
    return selected_samples


def generate_mixed_csv(output_csv, num_samples, num_sources, split, one_source_per_category=False):
    category_pools, bird_source_pools = load_category_pools(split)
    output_data = []

    for _ in range(num_samples):
        if one_source_per_category:
            selected_samples = sample_one_source_per_category(
                category_pools, bird_source_pools
            )
        else:
            while True:
                selected_samples = [
                    sample_one_source(category_pools, bird_source_pools)
                    for _ in range(num_sources)
                ]
                current_categories = [sample["category"] for sample in selected_samples]
                if len(set(current_categories)) > 1:
                    break

        row_data = []
        for sample in selected_samples:
            snr_db = random.uniform(-3.0, 3.0)
            row_data.extend([sample["audio_path"], sample["category"], snr_db])

        output_data.append(row_data)

    column_names = []
    for i in range(1, num_sources + 1):
        column_names.extend([f"s{i}_path", f"s{i}_label", f"s{i}_snr"])

    output_df = pd.DataFrame(output_data, columns=column_names)
    output_df.to_csv(output_csv, index=False)
    print(f"已生成 {num_samples} 条混合音频数据到 {output_csv}")
    print(f"数据形状: {output_df.shape}")
    if one_source_per_category:
        print("生成模式: speech/concert/bird 每类各一个源")


def export_bird_split_csvs(output_dir, random_seed=42):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bird_splits = build_bird_splits(random_seed=random_seed)
    for split in ("train", "valid", "test"):
        output_csv = output_dir / f"bird_{split}.csv"
        bird_splits[split].to_csv(output_csv, index=False)
        print(f"已导出 bird {split} CSV 到 {output_csv}")
        print(f"数据形状: {bird_splits[split].shape}")


def main():
    parser = argparse.ArgumentParser(description="生成三类混合音频数据的CSV文件")
    parser.add_argument("--output_csv", type=str, help="输出CSV文件路径")
    parser.add_argument(
        "--type",
        type=str,
        choices=["train", "valid", "test"],
        help="生成的数据集类型",
    )
    parser.add_argument(
        "--num_sources",
        type=int,
        choices=[2, 3, 4, 5],
        help="混合声源数量: 2, 3, 4 或 5",
    )
    parser.add_argument(
        "--export_bird_splits",
        action="store_true",
        help="导出 bird_train.csv、bird_valid.csv、bird_test.csv 后退出",
    )
    parser.add_argument(
        "--bird_output_dir",
        type=str,
        default=None,
        help="bird split CSV 输出目录；默认输出到 dataset/3class_speech_concert_bird",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="bird split 随机种子",
    )
    parser.add_argument(
        "--one_source_per_category",
        action="store_true",
        help="仅用于 3mix：每条样本固定包含 speech、concert、bird 各一个源",
    )

    args = parser.parse_args()

    if args.export_bird_splits:
        output_dir = args.bird_output_dir or DATA_ROOT
        export_bird_split_csvs(output_dir, random_seed=args.random_seed)
        return

    missing_args = [
        arg_name
        for arg_name, value in (
            ("--output_csv", args.output_csv),
            ("--type", args.type),
            ("--num_sources", args.num_sources),
        )
        if value is None
    ]
    if missing_args:
        parser.error(
            "生成混合 CSV 时需要参数: " + ", ".join(missing_args)
        )

    if args.one_source_per_category and args.num_sources != 3:
        parser.error("--one_source_per_category 只能与 --num_sources 3 一起使用")

    if args.type == "train":
        num_samples = 20000
    elif args.type == "valid":
        num_samples = 5000
    else:
        num_samples = 3000

    print(f"开始生成 {args.type} 数据，样本数: {num_samples}")
    generate_mixed_csv(
        output_csv=args.output_csv,
        num_samples=num_samples,
        num_sources=args.num_sources,
        split=args.type,
        one_source_per_category=args.one_source_per_category,
    )


if __name__ == "__main__":
    main()
