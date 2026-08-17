#!/usr/bin/env python3
import argparse
import csv
import os

import numpy as np
import scipy.io.wavfile as wav


CATEGORIES = ("speech", "concert", "bird")
EPS = 1e-12


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute global RMS(es)/RMS(gt) statistics for speech, concert, and bird "
            "by reading GT audio from a reference input directory and estimated audio "
            "from an inference output directory."
        )
    )
    parser.add_argument(
        "--inference_input_dir",
        type=str,
        required=True,
        help="Reference sample root directory containing GT wav files",
    )
    parser.add_argument(
        "--inference_output_dir",
        type=str,
        required=True,
        help="Inference result root directory containing sample subdirectories",
    )
    parser.add_argument(
        "--save_csv",
        type=str,
        default=None,
        help="Optional CSV path to save per-sample RMS ratio records",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print skip details for missing files and zero-energy GT audio",
    )
    return parser.parse_args()


def load_audio_float32(path):
    sample_rate, audio = wav.read(path)
    if np.issubdtype(audio.dtype, np.integer):
        max_value = max(abs(np.iinfo(audio.dtype).min), np.iinfo(audio.dtype).max)
        audio = audio.astype(np.float32) / float(max_value)
    else:
        audio = audio.astype(np.float32)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    return sample_rate, audio


def compute_rms(audio):
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "max": float(np.max(values)),
    }


def ratio_to_db(ratio):
    return 20.0 * np.log10(max(ratio, EPS))


def main():
    args = parse_args()

    if not os.path.isdir(args.inference_input_dir):
        raise FileNotFoundError(
            f"inference_input_dir does not exist: {args.inference_input_dir}"
        )

    if not os.path.isdir(args.inference_output_dir):
        raise FileNotFoundError(
            f"inference_output_dir does not exist: {args.inference_output_dir}"
        )

    sample_names = sorted(
        item
        for item in os.listdir(args.inference_output_dir)
        if os.path.isdir(os.path.join(args.inference_output_dir, item))
    )

    global_ratios = {category: [] for category in CATEGORIES}
    csv_rows = []
    total_samples = 0
    missing_input_sample_dirs = 0
    skipped_missing_files = {category: 0 for category in CATEGORIES}
    skipped_zero_gt = {category: 0 for category in CATEGORIES}
    skipped_empty_audio = {category: 0 for category in CATEGORIES}
    sample_rate_mismatch = {category: 0 for category in CATEGORIES}

    for sample_name in sample_names:
        input_sample_dir = os.path.join(args.inference_input_dir, sample_name)
        output_sample_dir = os.path.join(args.inference_output_dir, sample_name)
        total_samples += 1

        if not os.path.isdir(input_sample_dir):
            missing_input_sample_dirs += 1
            if args.verbose:
                print(f"[skip sample] {sample_name}: missing input sample dir")
            continue

        for category in CATEGORIES:
            gt_path = os.path.join(input_sample_dir, f"{category}_gt.wav")
            es_path = os.path.join(output_sample_dir, f"{category}_es.wav")

            if not os.path.exists(gt_path) or not os.path.exists(es_path):
                skipped_missing_files[category] += 1
                if args.verbose:
                    print(f"[skip {category}] {sample_name}: missing gt or es file")
                continue

            gt_sr, gt_audio = load_audio_float32(gt_path)
            es_sr, es_audio = load_audio_float32(es_path)

            if gt_sr != es_sr:
                sample_rate_mismatch[category] += 1
                if args.verbose:
                    print(
                        f"[warn {category}] {sample_name}: sample rate mismatch "
                        f"gt={gt_sr}, es={es_sr}, use min-length waveform only"
                    )

            min_len = min(len(gt_audio), len(es_audio))
            if min_len == 0:
                skipped_empty_audio[category] += 1
                if args.verbose:
                    print(f"[skip {category}] {sample_name}: empty gt/es audio")
                continue

            gt_audio = gt_audio[:min_len]
            es_audio = es_audio[:min_len]

            gt_rms = compute_rms(gt_audio)
            if gt_rms <= EPS:
                skipped_zero_gt[category] += 1
                if args.verbose:
                    print(f"[skip {category}] {sample_name}: gt rms is zero")
                continue

            es_rms = compute_rms(es_audio)
            ratio = es_rms / gt_rms
            ratio_db = ratio_to_db(ratio)

            global_ratios[category].append(ratio)
            csv_rows.append(
                {
                    "sample_name": sample_name,
                    "category": category,
                    "length_samples": min_len,
                    "sample_rate_hz": gt_sr,
                    "gt_rms": gt_rms,
                    "es_rms": es_rms,
                    "ratio_es_over_gt": ratio,
                    "ratio_db": ratio_db,
                }
            )

    print("=" * 72)
    print("ES/GT RMS Ratio Statistics")
    print("=" * 72)
    print(f"Reference input dir: {args.inference_input_dir}")
    print(f"Inference output dir: {args.inference_output_dir}")
    print(f"Output sample directories scanned: {total_samples}")
    print(f"Samples skipped for missing input sample dir: {missing_input_sample_dirs}")
    print()

    for category in CATEGORIES:
        print(f"[{category}]")
        print(f"  valid count:              {len(global_ratios[category])}")
        print(f"  missing gt/es file:       {skipped_missing_files[category]}")
        print(f"  zero-energy gt skipped:   {skipped_zero_gt[category]}")
        print(f"  empty-audio skipped:      {skipped_empty_audio[category]}")
        print(f"  sample-rate mismatch:     {sample_rate_mismatch[category]}")

        if global_ratios[category]:
            stats = summarize(global_ratios[category])
            mean_db = ratio_to_db(stats["mean"])
            median_db = ratio_to_db(stats["median"])
            min_db = ratio_to_db(stats["min"])
            max_db = ratio_to_db(stats["max"])
            print(f"  ratio mean db:            {mean_db:.3f} dB (linear {stats['mean']:.6f})")
            print(f"  ratio std linear:         {stats['std']:.6f}")
            print(f"  ratio median db:          {median_db:.3f} dB (linear {stats['median']:.6f})")
            print(f"  ratio min db:             {min_db:.3f} dB (linear {stats['min']:.6f})")
            print(f"  ratio max db:             {max_db:.3f} dB (linear {stats['max']:.6f})")
        else:
            print("  no valid samples")
        print()

    if args.save_csv:
        csv_dir = os.path.dirname(os.path.abspath(args.save_csv))
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)
        with open(args.save_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "sample_name",
                    "category",
                    "length_samples",
                    "sample_rate_hz",
                    "gt_rms",
                    "es_rms",
                    "ratio_es_over_gt",
                    "ratio_db",
                ],
            )
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"Saved per-sample records to: {args.save_csv}")


if __name__ == "__main__":
    main()
