import csv
import os
import re
import sys

import numpy as np
import scipy.io.wavfile as wav
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from DNN_models.Complex_MTASS_model import ComplexMTASSLightning
from DNN_models.Complex_MTASS import Complex_MTASS
from DNN_models.Complex_MTASS_Solver import Complex_MTASS_model


def sdr_cost(estimated, target, eps=1e-8):
    estimated = torch.from_numpy(estimated).float()
    target = torch.from_numpy(target).float()

    dot = torch.sum(estimated * target, dim=-1, keepdim=True)
    sign = torch.sign(dot)
    estimated = estimated * sign

    est_norm = torch.norm(estimated, dim=-1, keepdim=True) + eps
    tgt_norm = torch.norm(target, dim=-1, keepdim=True) + eps
    estimated = estimated * (tgt_norm / est_norm)

    signal_pow = torch.sum(target ** 2, dim=-1) + eps
    noise = estimated - target
    noise_pow = torch.sum(noise ** 2, dim=-1) + eps

    sdr = 10 * torch.log10(signal_pow / noise_pow)
    return sdr.item()


def sisdr_cost(estimated, target, eps=1e-8):
    estimated = torch.from_numpy(estimated).float()
    target = torch.from_numpy(target).float()

    dot = torch.sum(estimated * target, dim=-1, keepdim=True) + eps
    s_energy = torch.sum(target ** 2, dim=-1, keepdim=True) + eps

    scale = dot / s_energy
    target_scaled = scale * target
    e_noise = estimated - target_scaled
    target_pow = torch.sum(target_scaled ** 2, dim=-1) + eps
    noise_pow = torch.sum(e_noise ** 2, dim=-1) + eps

    sisdr = 10 * torch.log10(target_pow / noise_pow)
    return sisdr.item()


def inverse_stft(inputs, win_len, win_hop, fft_len, device):
    cutoff = fft_len // 2 + 1
    real_part = inputs[:, :cutoff, :]
    imag_part = inputs[:, cutoff:, :]

    complex_spec = torch.complex(real_part, imag_part)
    istft_window = torch.hamming_window(win_len, device=device)

    reconstruction = torch.istft(
        complex_spec,
        n_fft=fft_len,
        hop_length=win_hop,
        win_length=win_len,
        window=istft_window,
        center=False,
        normalized=False,
        onesided=True,
        return_complex=False,
    )

    return reconstruction


def stft(wav_data, win_len, win_hop, fft_len, device, debug=False):
    _, num_samples = wav_data.shape

    if num_samples <= win_len:
        nf = 1
    else:
        nf = (num_samples - win_len + win_hop - 1) // win_hop + 1
    pad_length = (nf - 1) * win_hop + win_len
    padding_size = pad_length - num_samples
    if padding_size > 0:
        wav_data = torch.nn.functional.pad(wav_data, (0, padding_size), mode='constant', value=0.0)

    window = torch.hamming_window(win_len, device=device)
    complex_spec = torch.stft(
        wav_data,
        n_fft=fft_len,
        hop_length=win_hop,
        win_length=win_len,
        window=window,
        center=False,
        normalized=False,
        onesided=True,
        return_complex=False,
    )
    if debug:
        print(f"[DEBUG STFT] input_shape={wav_data.shape}, complex_spec.shape={complex_spec.shape}")
    real_part = complex_spec[:, :, :, 0]
    imag_part = complex_spec[:, :, :, 1]
    features = torch.cat([real_part, imag_part], dim=1)
    if debug:
        print(f"[DEBUG STFT] output_shape={features.shape}")
    return features


def wav_write(data, path, filename, fs):
    full_path = os.path.join(path, filename)
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    wav.write(full_path, fs, (data * 32767).astype(np.int16))


def wav_read_float(wav_path):
    fs, audio_data = wav.read(wav_path)
    if audio_data.dtype != np.float32:
        if np.issubdtype(audio_data.dtype, np.integer):
            audio_data = audio_data.astype(np.float32) / np.iinfo(audio_data.dtype).max
        else:
            audio_data = audio_data.astype(np.float32)
    return fs, audio_data


def parse_csv_metadata(csv_path, categories):
    sample_category_counts = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, skipinitialspace=True)
        next(reader)
        for row in reader:
            counts = {category: 0 for category in categories}
            for i in range(0, len(row), 3):
                if i + 1 >= len(row):
                    continue
                label_lower = row[i + 1].lower()
                matched = False
                for category in categories:
                    if category in label_lower:
                        counts[category] += 1
                        matched = True
                        break
                if not matched:
                    raise ValueError(f"Unknown category label in csv: {row[i + 1]}")
            sample_category_counts.append(counts)

    return sample_category_counts


def get_sample_index(sample_name):
    match = re.fullmatch(r'sample(\d+)', sample_name)
    if match is None:
        raise ValueError(
            f"Sample directory name '{sample_name}' does not match expected format 'sample{{idx}}'"
        )
    return int(match.group(1))


def collect_sample_dirs(wav_dir, num_samples=None):
    sample_dirs = []
    for item in sorted(os.listdir(wav_dir)):
        item_path = os.path.join(wav_dir, item)
        if os.path.isdir(item_path):
            mixture_path = os.path.join(item_path, 'mixture.wav')
            if os.path.exists(mixture_path):
                sample_dirs.append(item_path)

    if num_samples is not None:
        sample_dirs = sample_dirs[:num_samples]
    return sample_dirs


def get_existing_classes(sample_dir, categories, gt_filename_map):
    existing_classes = []
    gt_paths = {}
    eps = 1e-8

    for category in categories:
        gt_path = os.path.join(sample_dir, gt_filename_map[category])
        if os.path.exists(gt_path):
            fs, gt_data = wav.read(gt_path)
            if gt_data.dtype != np.float32:
                gt_data = gt_data.astype(np.float32) / 32767.0
            if np.sum(gt_data ** 2) > eps:
                existing_classes.append(category)
                gt_paths[category] = (fs, gt_data)

    return existing_classes, gt_paths


def print_category_stats(label, sdr_values, sisdr_values, sdri_values=None):
    if sdr_values:
        print(f"{label} SDR:    {np.mean(sdr_values):.2f} +/- {np.std(sdr_values):.2f}")
        print(f"{label} SI-SDR: {np.mean(sisdr_values):.2f} +/- {np.std(sisdr_values):.2f}")
        if sdri_values:
            print(f"{label} SDRi:   {np.mean(sdri_values):.2f} +/- {np.std(sdri_values):.2f}")


def print_bucket_stats(title, sdr_values, sisdr_values, sdri_values=None):
    if sdr_values:
        print(f"{title} SDR:     {np.mean(sdr_values):.2f} +/- {np.std(sdr_values):.2f}")
        print(f"{title} SI-SDR:  {np.mean(sisdr_values):.2f} +/- {np.std(sisdr_values):.2f}")
        if sdri_values:
            print(f"{title} SDRi:    {np.mean(sdri_values):.2f} +/- {np.std(sdri_values):.2f}")


def load_offline_model(ckpt_path, device):
    model = ComplexMTASSLightning.load_from_checkpoint(
        ckpt_path,
        map_location=device,
        model_class=Complex_MTASS,
        loss_class=Complex_MTASS_model,
    )
    model.to(device)
    model.eval()
    model.freeze()
    return model


def resolve_device(device_name):
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but is not available")
        return torch.device("cuda")
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raise ValueError(f"Unsupported device: {device_name}")


def process_offline(model, mixture_path, existing_classes, categories, device, win_len=512, win_inc=256, fft_len=512, debug=False):
    fs_read, audio_data = wav_read_float(mixture_path)

    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)

    audio_tensor = torch.from_numpy(audio_data).float().to(device)
    audio_tensor = audio_tensor.unsqueeze(0)

    if debug:
        print(
            f"[DEBUG process] mixture_path={mixture_path}, "
            f"audio_data.shape={audio_data.shape}, tensor.shape={audio_tensor.shape}"
        )

    x1 = stft(audio_tensor, win_len, win_inc, fft_len, device, debug=debug)

    with torch.no_grad():
        z1, z2, z3 = model(x1)

    results = {}
    for category, output in zip(categories, [z1, z2, z3]):
        if category in existing_classes:
            est_wav = inverse_stft(output, win_len, win_inc, fft_len, device)
            results[category] = est_wav.squeeze().detach().cpu().numpy()

    mixture_wav = inverse_stft(x1, win_len, win_inc, fft_len, device)
    results['mixture'] = mixture_wav.squeeze().detach().cpu().numpy()

    return results, fs_read


def run_model_in_chunks(model, inputs, chunk_frames, debug=False):
    if chunk_frames <= 0:
        raise ValueError(f"chunk_frames must be positive, got {chunk_frames}")

    num_frames = inputs.shape[-1]
    z1_chunks = []
    z2_chunks = []
    z3_chunks = []

    with torch.no_grad():
        for start in range(0, num_frames, chunk_frames):
            end = min(start + chunk_frames, num_frames)
            input_chunk = inputs[:, :, start:end]
            if debug:
                print(f"[DEBUG chunk] input frames {start}:{end}, shape={input_chunk.shape}")
            z1_chunk, z2_chunk, z3_chunk = model(input_chunk)
            expected_frames = end - start
            for name, output_chunk in zip(("z1", "z2", "z3"), (z1_chunk, z2_chunk, z3_chunk)):
                if output_chunk.shape[-1] != expected_frames:
                    raise ValueError(
                        f"Model output {name} has {output_chunk.shape[-1]} frames for "
                        f"{expected_frames} input frames in chunk {start}:{end}"
                    )
            z1_chunks.append(z1_chunk)
            z2_chunks.append(z2_chunk)
            z3_chunks.append(z3_chunk)

    return (
        torch.cat(z1_chunks, dim=-1),
        torch.cat(z2_chunks, dim=-1),
        torch.cat(z3_chunks, dim=-1),
    )


def process_offline_chunked_model(
    model,
    mixture_path,
    existing_classes,
    categories,
    device,
    win_len=512,
    win_inc=256,
    fft_len=512,
    chunk_frames=100,
    debug=False,
):
    fs_read, audio_data = wav_read_float(mixture_path)

    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)

    audio_tensor = torch.from_numpy(audio_data).float().to(device)
    audio_tensor = audio_tensor.unsqueeze(0)

    if debug:
        print(
            f"[DEBUG process chunked] mixture_path={mixture_path}, "
            f"audio_data.shape={audio_data.shape}, tensor.shape={audio_tensor.shape}"
        )

    x1 = stft(audio_tensor, win_len, win_inc, fft_len, device, debug=debug)

    if debug:
        num_frames = x1.shape[-1]
        num_chunks = (num_frames + chunk_frames - 1) // chunk_frames
        print(
            f"[DEBUG process chunked] spec_shape={x1.shape}, "
            f"chunk_frames={chunk_frames}, num_chunks={num_chunks}"
        )

    z1, z2, z3 = run_model_in_chunks(model, x1, chunk_frames, debug=debug)

    results = {}
    for category, output in zip(categories, [z1, z2, z3]):
        if category in existing_classes:
            est_wav = inverse_stft(output, win_len, win_inc, fft_len, device)
            results[category] = est_wav.squeeze().detach().cpu().numpy()

    mixture_wav = inverse_stft(x1, win_len, win_inc, fft_len, device)
    results['mixture'] = mixture_wav.squeeze().detach().cpu().numpy()

    return results, fs_read
