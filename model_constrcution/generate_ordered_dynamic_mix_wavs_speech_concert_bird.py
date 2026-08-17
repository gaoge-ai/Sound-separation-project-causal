#!/usr/bin/env python3
import argparse
import csv
import os
import random
from dataclasses import dataclass


try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x

TARGET_SAMPLE_RATE = 16000
SEGMENT_SECONDS = 10
SEGMENT_LENGTH = TARGET_SAMPLE_RATE * SEGMENT_SECONDS
CATEGORIES = ("speech", "concert", "bird")
CATEGORY_FILES = {
    "speech": "speech_gt.wav",
    "concert": "concert_gt.wav",
    "bird": "bird_gt.wav",
}


@dataclass
class SourceInfo:
    path: str
    label: str
    snr: str


@dataclass
class SegmentSpec:
    parent_mix_type: str
    parent_sample_idx: int
    source_trial_idx: int
    segment_idx: int
    mix_level: int
    order: tuple
    active_categories: tuple
    sources_by_category: dict


@dataclass
class SerialSpec:
    mix_type: str
    sample_idx: int
    source_trial_idx: int
    order: tuple
    levels: tuple
    sources_by_category: dict


def load_wav(path, target_sample_rate=TARGET_SAMPLE_RATE, max_length=SEGMENT_LENGTH):
    import librosa
    import numpy as np

    wav_data, _ = librosa.core.load(path, sr=target_sample_rate)
    if len(wav_data) > max_length:
        wav_data = wav_data[0:max_length]
    if len(wav_data) < max_length:
        wav_data = np.pad(wav_data, (0, max_length - len(wav_data)), "constant")
    return wav_data


def mix_audios(audios, snrs):
    import numpy as np

    target = audios[0]
    target_energy = np.sum(target ** 2)
    mixed = target.copy()
    scaled_audios = [target]

    for i in range(1, len(audios)):
        noise = audios[i]
        noise_energy = np.sum(noise ** 2)
        snr_db = float(snrs[i])
        snr_linear = 10 ** (snr_db / 10)
        scale = np.sqrt((target_energy / snr_linear) / (noise_energy + 1e-8))
        scaled_noise = noise * scale
        mixed += scaled_noise
        scaled_audios.append(scaled_noise)

    max_value = np.max(np.abs(mixed))
    if max_value > 1:
        mixed *= 0.9 / max_value
        scaled_audios = [audio * 0.9 / max_value for audio in scaled_audios]

    return mixed, scaled_audios


def label_to_category(label):
    label_lower = label.strip().lower()
    for category in CATEGORIES:
        if category in label_lower:
            return category
    raise ValueError(f"Unknown category label: {label}")


def parse_csv_by_category(csv_path):
    trials = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")

        required_columns = []
        for idx in range(1, 4):
            required_columns.extend([f"s{idx}_path", f"s{idx}_label", f"s{idx}_snr"])
        missing_columns = [
            column for column in required_columns if column not in reader.fieldnames
        ]
        if missing_columns:
            raise ValueError(
                f"CSV missing required columns: {', '.join(missing_columns)}"
            )

        for row_idx, row in enumerate(reader):
            sources_by_category = {}
            for source_idx in range(1, 4):
                label = row[f"s{source_idx}_label"].strip()
                category = label_to_category(label)
                if category in sources_by_category:
                    raise ValueError(
                        f"Row {row_idx} has duplicate category {category}: {csv_path}"
                    )
                sources_by_category[category] = SourceInfo(
                    path=row[f"s{source_idx}_path"].strip(),
                    label=label,
                    snr=row[f"s{source_idx}_snr"].strip(),
                )

            missing_categories = [
                category for category in CATEGORIES if category not in sources_by_category
            ]
            if missing_categories:
                raise ValueError(
                    f"Row {row_idx} missing categories: {', '.join(missing_categories)}"
                )

            trials.append(
                {
                    "source_trial_idx": row_idx,
                    "sources_by_category": sources_by_category,
                }
            )

    return trials


def sample_trials(trials, num_trials, seed):
    if num_trials <= 0:
        raise ValueError("--num_trials must be a positive integer")
    if num_trials % 2 != 0:
        raise ValueError("--num_trials must be even so the two orders split evenly")
    if num_trials > len(trials):
        raise ValueError(
            f"--num_trials={num_trials} exceeds available trials: {len(trials)}"
        )
    rng = random.Random(seed)
    return rng.sample(trials, num_trials)


def split_half_orders(num_trials, first_order, second_order):
    first_count = num_trials // 2
    second_count = num_trials - first_count
    return [first_order] * first_count + [second_order] * second_count


def build_specs(sampled_trials):
    serial_specs = []
    segment_specs_by_level = {1: [], 2: [], 3: []}

    one_two_three_orders = split_half_orders(
        len(sampled_trials),
        ("speech", "concert", "bird"),
        ("concert", "speech", "bird"),
    )
    three_two_one_orders = split_half_orders(
        len(sampled_trials),
        ("speech", "concert", "bird"),
        ("concert", "speech", "bird"),
    )

    spec_inputs = [
        ("1_2_3_mix", (1, 2, 3), one_two_three_orders),
        ("3_2_1_mix", (3, 2, 1), three_two_one_orders),
    ]

    for mix_type, levels, orders in spec_inputs:
        for sample_idx, (trial, order) in enumerate(zip(sampled_trials, orders)):
            serial_spec = SerialSpec(
                mix_type=mix_type,
                sample_idx=sample_idx,
                source_trial_idx=trial["source_trial_idx"],
                order=order,
                levels=levels,
                sources_by_category=trial["sources_by_category"],
            )
            serial_specs.append(serial_spec)

            for segment_idx, mix_level in enumerate(levels):
                active_categories = tuple(order[:mix_level])
                segment_spec = SegmentSpec(
                    parent_mix_type=mix_type,
                    parent_sample_idx=sample_idx,
                    source_trial_idx=trial["source_trial_idx"],
                    segment_idx=segment_idx,
                    mix_level=mix_level,
                    order=order,
                    active_categories=active_categories,
                    sources_by_category=trial["sources_by_category"],
                )
                segment_specs_by_level[mix_level].append(segment_spec)

    return serial_specs, segment_specs_by_level


def load_trial_audios(sources_by_category):
    audios_by_category = {}
    for category in CATEGORIES:
        source = sources_by_category[category]
        if not os.path.exists(source.path):
            raise FileNotFoundError(f"File not found: {source.path}")
        audios_by_category[category] = load_wav(source.path)
    return audios_by_category


def scale_trial_sources(order, sources_by_category, audios_by_category):
    audios = [audios_by_category[category] for category in order]
    snrs = [sources_by_category[category].snr for category in order]
    _, scaled_sources = mix_audios(audios, snrs)
    return {
        category: scaled_source
        for category, scaled_source in zip(order, scaled_sources)
    }


def make_segment(active_categories, scaled_audios_by_category):
    import numpy as np

    category_wavs = {
        category: np.zeros_like(next(iter(scaled_audios_by_category.values())))
        for category in CATEGORIES
    }
    for category in active_categories:
        category_wavs[category] += scaled_audios_by_category[category]

    mixed_wav = np.zeros_like(next(iter(scaled_audios_by_category.values())))
    for category in active_categories:
        mixed_wav += category_wavs[category]
    return mixed_wav, category_wavs


def make_serial_trial(spec):
    import numpy as np

    audios_by_category = load_trial_audios(spec.sources_by_category)
    scaled_audios_by_category = scale_trial_sources(
        order=spec.order,
        sources_by_category=spec.sources_by_category,
        audios_by_category=audios_by_category,
    )
    mixture_segments = []
    category_segments = {category: [] for category in CATEGORIES}

    for mix_level in spec.levels:
        active_categories = spec.order[:mix_level]
        mixed_wav, category_wavs = make_segment(
            active_categories=active_categories,
            scaled_audios_by_category=scaled_audios_by_category,
        )
        mixture_segments.append(mixed_wav)
        for category in CATEGORIES:
            category_segments[category].append(category_wavs[category])

    mixture = np.concatenate(mixture_segments)
    category_wavs = {
        category: np.concatenate(category_segments[category])
        for category in CATEGORIES
    }
    return mixture, category_wavs


def make_single_segment_trial(spec):
    audios_by_category = load_trial_audios(spec.sources_by_category)
    scaled_audios_by_category = scale_trial_sources(
        order=spec.order,
        sources_by_category=spec.sources_by_category,
        audios_by_category=audios_by_category,
    )
    return make_segment(
        active_categories=spec.active_categories,
        scaled_audios_by_category=scaled_audios_by_category,
    )


def float_to_int16(wav_data):
    import numpy as np

    return (wav_data * 32767).astype(np.int16)


def write_sample_dir(sample_dir, mixture, category_wavs):
    import scipy.io.wavfile as wav

    os.makedirs(sample_dir, exist_ok=True)
    wav.write(
        os.path.join(sample_dir, "mixture.wav"),
        TARGET_SAMPLE_RATE,
        float_to_int16(mixture),
    )

    for category, filename in CATEGORY_FILES.items():
        wav.write(
            os.path.join(sample_dir, filename),
            TARGET_SAMPLE_RATE,
            float_to_int16(category_wavs[category]),
        )


def append_source_columns(row, active_categories, sources_by_category):
    for idx, category in enumerate(active_categories, start=1):
        source = sources_by_category[category]
        row[f"s{idx}_path"] = source.path
        row[f"s{idx}_label"] = source.label
        row[f"s{idx}_snr"] = source.snr
    return row


def source_columns(num_sources):
    columns = []
    for idx in range(1, num_sources + 1):
        columns.extend([f"s{idx}_path", f"s{idx}_label", f"s{idx}_snr"])
    return columns


def write_serial_csv(csv_path, specs):
    fieldnames = [
        "sample_name",
        "mix_type",
        "source_trial_idx",
        "order",
        "levels",
    ] + source_columns(3)

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for spec in specs:
            row = {
                "sample_name": f"sample{spec.sample_idx}",
                "mix_type": spec.mix_type,
                "source_trial_idx": spec.source_trial_idx,
                "order": "-".join(spec.order),
                "levels": "-".join(str(level) for level in spec.levels),
            }
            row = append_source_columns(row, spec.order, spec.sources_by_category)
            writer.writerow(row)


def write_segment_csv(csv_path, specs, mix_level):
    fieldnames = [
        "sample_name",
        "parent_mix_type",
        "parent_sample_name",
        "source_trial_idx",
        "segment_idx",
        "mix_level",
        "order",
        "active_categories",
    ] + source_columns(mix_level)

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sample_idx, spec in enumerate(specs):
            row = {
                "sample_name": f"sample{sample_idx}",
                "parent_mix_type": spec.parent_mix_type,
                "parent_sample_name": f"sample{spec.parent_sample_idx}",
                "source_trial_idx": spec.source_trial_idx,
                "segment_idx": spec.segment_idx,
                "mix_level": spec.mix_level,
                "order": "-".join(spec.order),
                "active_categories": "-".join(spec.active_categories),
            }
            row = append_source_columns(
                row, spec.active_categories, spec.sources_by_category
            )
            writer.writerow(row)


def generate_serial_wavs(serial_specs, output_dir):
    specs_by_type = {
        "1_2_3_mix": [spec for spec in serial_specs if spec.mix_type == "1_2_3_mix"],
        "3_2_1_mix": [spec for spec in serial_specs if spec.mix_type == "3_2_1_mix"],
    }

    for mix_type, specs in specs_by_type.items():
        wav_dir = os.path.join(output_dir, mix_type)
        os.makedirs(wav_dir, exist_ok=True)
        print(f"Generating {len(specs)} {mix_type} wav samples...")
        for spec in tqdm(specs):
            mixture, category_wavs = make_serial_trial(spec)
            sample_dir = os.path.join(wav_dir, f"sample{spec.sample_idx}")
            write_sample_dir(sample_dir, mixture, category_wavs)


def generate_segment_wavs(segment_specs_by_level, output_dir):
    for mix_level in (1, 2, 3):
        specs = segment_specs_by_level[mix_level]
        wav_dir = os.path.join(output_dir, f"{mix_level}mix")
        os.makedirs(wav_dir, exist_ok=True)
        print(f"Generating {len(specs)} {mix_level}mix wav samples...")
        for sample_idx, spec in enumerate(tqdm(specs)):
            mixture, category_wavs = make_single_segment_trial(spec)
            sample_dir = os.path.join(wav_dir, f"sample{sample_idx}")
            write_sample_dir(sample_dir, mixture, category_wavs)


def write_metadata(serial_specs, segment_specs_by_level, output_dir):
    metadata_dir = os.path.join(output_dir, "metadata")
    os.makedirs(metadata_dir, exist_ok=True)

    write_serial_csv(
        os.path.join(metadata_dir, "ordered_1_2_3_mix.csv"),
        [spec for spec in serial_specs if spec.mix_type == "1_2_3_mix"],
    )
    write_serial_csv(
        os.path.join(metadata_dir, "ordered_3_2_1_mix.csv"),
        [spec for spec in serial_specs if spec.mix_type == "3_2_1_mix"],
    )

    for mix_level in (1, 2, 3):
        write_segment_csv(
            os.path.join(metadata_dir, f"ordered_{mix_level}mix.csv"),
            segment_specs_by_level[mix_level],
            mix_level,
        )


def generate_ordered_dynamic_mixes(csv_path, output_dir, num_trials, seed):
    os.makedirs(output_dir, exist_ok=True)
    trials = parse_csv_by_category(csv_path)
    sampled_trials = sample_trials(trials, num_trials, seed)
    serial_specs, segment_specs_by_level = build_specs(sampled_trials)

    write_metadata(serial_specs, segment_specs_by_level, output_dir)
    generate_serial_wavs(serial_specs, output_dir)
    generate_segment_wavs(segment_specs_by_level, output_dir)

    print(f"Generated ordered dynamic mix wavs to: {output_dir}")
    print(f"Serial mix samples: {len(serial_specs)}")
    for mix_level in (1, 2, 3):
        print(f"{mix_level}mix segment samples: {len(segment_specs_by_level[mix_level])}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate ordered 1-2-3 / 3-2-1 serial mixes and standalone "
            "1mix/2mix/3mix wav trials for speech/concert/bird."
        )
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default=(
            "../dataset/3class_speech_concert_bird_20260424/metadata/"
            "test_3mix_each_class.csv"
        ),
        help="Input 3mix CSV with one speech, one concert, and one bird source per row",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./ordered_dynamic_mix_wavs_speech_concert_bird",
        help="Output directory for wavs and metadata",
    )
    parser.add_argument(
        "--num_trials",
        type=int,
        default=20,
        help="Number of source CSV trials to sample",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260508,
        help="Random seed used for source trial sampling",
    )

    args = parser.parse_args()
    generate_ordered_dynamic_mixes(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        num_trials=args.num_trials,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
