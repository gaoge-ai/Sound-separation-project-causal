#!/usr/bin/env python3
import argparse
import os

import numpy as np
from tqdm import tqdm

from wav_offline_eval_core import (
    collect_sample_dirs,
    get_existing_classes,
    get_sample_index,
    load_offline_model,
    parse_csv_metadata,
    print_bucket_stats,
    print_category_stats,
    process_offline_chunked_model,
    resolve_device,
    sdr_cost,
    sisdr_cost,
    wav_read_float,
    wav_write,
)


def build_output_sample_name(sample_name, existing_classes, rename_output):
    if rename_output and existing_classes:
        classes_suffix = '-'.join(existing_classes)
        return f"{sample_name}_{classes_suffix}"
    return sample_name


def load_existing_estimates(output_sample_dir, existing_classes, est_filename_map):
    estimates = {}
    missing_categories = []

    for category in existing_classes:
        estimate_path = os.path.join(output_sample_dir, est_filename_map[category])
        if not os.path.exists(estimate_path):
            missing_categories.append(category)
            continue
        _, estimate = wav_read_float(estimate_path)
        if len(estimate.shape) > 1:
            estimate = np.mean(estimate, axis=1)
        estimates[category] = estimate

    return estimates, missing_categories


def load_mixture_audio(mixture_path):
    _, mixture = wav_read_float(mixture_path)
    if len(mixture.shape) > 1:
        mixture = np.mean(mixture, axis=1)
    return mixture


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wav_dir', type=str, required=True, help='Directory with test wav samples')
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to model checkpoint .ckpt')
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./wav_offline_results_speech_concert_bird_chunk_model',
        help='Folder to save results',
    )
    parser.add_argument(
        '--chunk_frames',
        type=int,
        default=100,
        help='Number of STFT frames to feed to the model per non-overlapping chunk',
    )
    parser.add_argument('--debug', action='store_true', help='Print chunked inference debug information')
    parser.add_argument('--csv_path', type=str, default=None, help='CSV metadata path generated for the wav samples')
    parser.add_argument(
        '--mode',
        type=str,
        default='auto',
        choices=['auto', 'infer_and_eval', 'eval_only'],
        help='auto: reuse existing outputs when complete; eval_only: never infer; infer_and_eval: always infer',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='Device for model inference',
    )
    parser.add_argument('--num_samples', type=int, default=None, help='Number of samples to test')
    parser.add_argument('--rename_output', action='store_true', default=True, help='Rename output dir with classes suffix')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    categories = ['speech', 'concert', 'bird']
    gt_filename_map = {
        'speech': 'speech_gt.wav',
        'concert': 'concert_gt.wav',
        'bird': 'bird_gt.wav',
    }
    est_filename_map = {
        'speech': 'speech_es.wav',
        'concert': 'concert_es.wav',
        'bird': 'bird_es.wav',
    }

    device = resolve_device(args.device)
    print(f"Using device: {device}")
    print(f"Mode: {args.mode}")
    print(f"Chunk frames: {args.chunk_frames}")

    sample_dirs = collect_sample_dirs(args.wav_dir, args.num_samples)
    print(f"Found {len(sample_dirs)} samples")

    sample_category_counts = None
    if args.csv_path is not None:
        sample_category_counts = parse_csv_metadata(args.csv_path, categories)

    all_metrics = {category: {'sdr': [], 'sisdr': [], 'sdri': []} for category in categories}
    bucket_metrics = None
    if sample_category_counts is not None:
        bucket_metrics = {}
        for category in categories:
            bucket_metrics[f'{category}_single'] = {'sdr': [], 'sisdr': [], 'sdri': []}
            bucket_metrics[f'{category}_multi'] = {'sdr': [], 'sisdr': [], 'sdri': []}

    win_len = 512
    win_inc = 256
    fft_len = 512
    model = None

    for sample_dir in tqdm(sample_dirs, desc="Processing samples"):
        sample_name = os.path.basename(sample_dir)
        mixture_path = os.path.join(sample_dir, 'mixture.wav')
        mixture = load_mixture_audio(mixture_path)

        existing_classes, gt_paths = get_existing_classes(sample_dir, categories, gt_filename_map)
        output_sample_name = build_output_sample_name(sample_name, existing_classes, args.rename_output)
        output_sample_dir = os.path.join(args.output_dir, output_sample_name)
        os.makedirs(output_sample_dir, exist_ok=True)

        category_counts = None
        if sample_category_counts is not None:
            csv_sample_idx = get_sample_index(sample_name)
            if csv_sample_idx >= len(sample_category_counts):
                raise IndexError(
                    f"Sample index {csv_sample_idx} from '{sample_name}' exceeds csv size {len(sample_category_counts)}"
                )
            category_counts = sample_category_counts[csv_sample_idx]

        estimates, missing_categories = load_existing_estimates(output_sample_dir, existing_classes, est_filename_map)
        should_infer = args.mode == 'infer_and_eval' or (args.mode == 'auto' and bool(missing_categories))

        if args.mode == 'eval_only' and missing_categories:
            missing_labels = ', '.join(missing_categories)
            raise FileNotFoundError(
                f"Missing estimated wavs for {sample_name} in eval_only mode: {missing_labels}"
            )

        fs = gt_paths[existing_classes[0]][0] if existing_classes else 16000

        if should_infer:
            if model is None:
                print("Loading model...")
                model = load_offline_model(args.ckpt_path, device)
                print("Model loaded!")

            results, fs = process_offline_chunked_model(
                model,
                mixture_path,
                existing_classes,
                categories,
                device,
                win_len,
                win_inc,
                fft_len,
                chunk_frames=args.chunk_frames,
                debug=args.debug,
            )

            for category in existing_classes:
                if category in results:
                    estimates[category] = results[category]
                    wav_write(results[category], output_sample_dir, est_filename_map[category], fs)

            wav_write(results['mixture'], output_sample_dir, 'mixture.wav', fs)

        for category in categories:
            if category in existing_classes and category in estimates:
                estimate = estimates[category]
                _, target = gt_paths[category]
                min_len = min(len(estimate), len(target))
                category_sdr = sdr_cost(estimate[:min_len], target[:min_len])
                category_sisdr = sisdr_cost(estimate[:min_len], target[:min_len])
                mixture_sdr = sdr_cost(mixture[:min_len], target[:min_len])
                category_sdri = category_sdr - mixture_sdr
                all_metrics[category]['sdr'].append(category_sdr)
                all_metrics[category]['sisdr'].append(category_sisdr)
                all_metrics[category]['sdri'].append(category_sdri)
                wav_write(estimate, output_sample_dir, est_filename_map[category], fs)

                if category_counts is not None and category_counts[category] >= 1:
                    bucket_name = f"{category}_single" if category_counts[category] == 1 else f"{category}_multi"
                    bucket_metrics[bucket_name]['sdr'].append(category_sdr)
                    bucket_metrics[bucket_name]['sisdr'].append(category_sisdr)
                    bucket_metrics[bucket_name]['sdri'].append(category_sdri)

    print("\n" + "=" * 60)
    print("SDR Statistics (Offline STFT/ISTFT, Chunked Model, Speech/Concert/Bird):")
    print("=" * 60)

    for category in categories:
        print_category_stats(
            category.capitalize(),
            all_metrics[category]['sdr'],
            all_metrics[category]['sisdr'],
            all_metrics[category]['sdri'],
        )

    if bucket_metrics is not None:
        print("\nCategory Count Breakdown:")
        for category in categories:
            print_bucket_stats(
                f"{category.capitalize()} Single-Source",
                bucket_metrics[f'{category}_single']['sdr'],
                bucket_metrics[f'{category}_single']['sisdr'],
                bucket_metrics[f'{category}_single']['sdri'],
            )
            print_bucket_stats(
                f"{category.capitalize()} Multi-Source",
                bucket_metrics[f'{category}_multi']['sdr'],
                bucket_metrics[f'{category}_multi']['sisdr'],
                bucket_metrics[f'{category}_multi']['sdri'],
            )

    all_sdr = []
    all_sisdr = []
    all_sdri = []
    for category in categories:
        all_sdr.extend(all_metrics[category]['sdr'])
        all_sisdr.extend(all_metrics[category]['sisdr'])
        all_sdri.extend(all_metrics[category]['sdri'])

    if all_sdr:
        print(f"\nTotal Average SDR:    {np.mean(all_sdr):.2f} +/- {np.std(all_sdr):.2f}")
        print(f"Total Average SI-SDR: {np.mean(all_sisdr):.2f} +/- {np.std(all_sisdr):.2f}")
        print(f"Total Average SDRi:   {np.mean(all_sdri):.2f} +/- {np.std(all_sdri):.2f}")

    print("=" * 60)


if __name__ == '__main__':
    main()
