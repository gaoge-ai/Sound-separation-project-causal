#!/usr/bin/env python3
import argparse
import os
import shutil


GT_FILENAMES = [
    "speech_gt.wav",
    "concert_gt.wav",
    "bird_gt.wav",
]

ESTIMATE_FILENAMES = [
    "speech_es.wav",
    "concert_es.wav",
    "bird_es.wav",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Copy GT wav files from the inference input directory into the "
            "matching inference output directories after separation has finished."
        )
    )
    parser.add_argument(
        "--inference_input_dir",
        type=str,
        required=True,
        help=(
            "The same sample directory root used as input to inference. "
            "This is used to locate the accurate GT paths."
        ),
    )
    parser.add_argument(
        "--inference_output_dir",
        type=str,
        required=True,
        help="Directory containing existing inference result sample folders",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite GT files if they already exist in the output directory",
    )
    parser.add_argument(
        "--copy_mixture",
        action="store_true",
        help="Also copy mixture.wav into the output directory",
    )
    return parser.parse_args()


def copy_file(src_path, dst_path, overwrite=False):
    if not os.path.exists(src_path):
        return "missing_src"

    if os.path.exists(dst_path) and not overwrite:
        return "exists"

    shutil.copy2(src_path, dst_path)
    return "copied"


def has_estimated_outputs(output_sample_dir):
    return any(
        os.path.exists(os.path.join(output_sample_dir, filename))
        for filename in ESTIMATE_FILENAMES
    )


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

    total_samples = 0
    eligible_samples = 0
    copied_files = 0
    existing_files = 0
    missing_source_files = 0
    missing_sample_dirs = 0
    skipped_without_estimates = 0

    filenames_to_copy = list(GT_FILENAMES)
    if args.copy_mixture:
        filenames_to_copy.append("mixture.wav")

    for sample_name in sample_names:
        output_sample_dir = os.path.join(args.inference_output_dir, sample_name)
        source_sample_dir = os.path.join(args.inference_input_dir, sample_name)
        total_samples += 1

        if not has_estimated_outputs(output_sample_dir):
            skipped_without_estimates += 1
            print(f"[skip no estimates] {sample_name}")
            continue

        eligible_samples += 1

        if not os.path.isdir(source_sample_dir):
            missing_sample_dirs += 1
            print(f"[missing sample] {sample_name}: source folder not found")
            continue

        for filename in filenames_to_copy:
            src_path = os.path.join(source_sample_dir, filename)
            dst_path = os.path.join(output_sample_dir, filename)
            result = copy_file(src_path, dst_path, overwrite=args.overwrite)

            if result == "copied":
                copied_files += 1
                print(f"[copied] {sample_name}/{filename}")
            elif result == "exists":
                existing_files += 1
                print(f"[skip exists] {sample_name}/{filename}")
            else:
                missing_source_files += 1
                print(f"[missing file] {sample_name}/{filename}")

    print("\n" + "=" * 60)
    print("Copy GT Summary")
    print("=" * 60)
    print(f"Samples scanned:            {total_samples}")
    print(f"Samples with estimates:     {eligible_samples}")
    print(f"Samples skipped no output:  {skipped_without_estimates}")
    print(f"Files copied:               {copied_files}")
    print(f"Files already existing:     {existing_files}")
    print(f"Missing source files:       {missing_source_files}")
    print(f"Missing sample folders:     {missing_sample_dirs}")
    print("=" * 60)


if __name__ == "__main__":
    main()
