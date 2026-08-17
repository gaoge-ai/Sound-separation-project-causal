#!/usr/bin/env python3
import argparse
import os

import numpy as np
import scipy.io.wavfile as wav
import torch
from tqdm import tqdm


def wav_read_float(path):
    fs, audio = wav.read(path)
    if audio.dtype != np.float32:
        if np.issubdtype(audio.dtype, np.integer):
            audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
        else:
            audio = audio.astype(np.float32)
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    return fs, audio


def wav_write_float(path, fs, audio):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    audio = np.asarray(audio, dtype=np.float32)
    audio = np.clip(audio, -1.0, 1.0)
    wav.write(path, fs, (audio * 32767).astype(np.int16))


def sisdr_cost(estimated, target, eps=1e-8):
    estimated = torch.from_numpy(estimated).float()
    target = torch.from_numpy(target).float()

    dot = torch.sum(estimated * target, dim=-1, keepdim=True) + eps
    target_energy = torch.sum(target ** 2, dim=-1, keepdim=True) + eps
    target_scaled = dot / target_energy * target
    noise = estimated - target_scaled

    target_pow = torch.sum(target_scaled ** 2, dim=-1) + eps
    noise_pow = torch.sum(noise ** 2, dim=-1) + eps
    return (10 * torch.log10(target_pow / noise_pow)).item()


def error_stats(estimated, target):
    min_len = min(len(estimated), len(target))
    estimated = estimated[:min_len]
    target = target[:min_len]
    diff = estimated - target
    return {
        "min_len": min_len,
        "sisdr": sisdr_cost(estimated, target),
        "rms_error": float(np.sqrt(np.mean(diff ** 2) + 1e-12)),
        "max_abs_error": float(np.max(np.abs(diff))),
    }


class StreamingSTFT:
    def __init__(self, win_len=512, win_inc=256, fft_len=512, device="cpu"):
        self.win_len = win_len
        self.win_inc = win_inc
        self.fft_len = fft_len
        self.device = device
        self.window = torch.hamming_window(win_len, device=device)
        self.input_buffer = torch.zeros(win_len, device=device)
        self.seen_first_frame = False

    def reset(self):
        self.input_buffer.zero_()
        self.seen_first_frame = False

    def process(self, new_samples):
        new_samples = torch.from_numpy(new_samples).float().to(self.device)
        num_new = len(new_samples)
        frames = []

        if not self.seen_first_frame and num_new >= self.win_len:
            self.input_buffer = new_samples[:self.win_len].clone()
            frames.append(torch.fft.rfft(self.input_buffer * self.window, n=self.fft_len))
            self.seen_first_frame = True
            remaining_start = self.win_len
        else:
            remaining_start = 0

        for start in range(remaining_start, num_new, self.win_inc):
            chunk = new_samples[start:min(start + self.win_inc, num_new)]
            if len(chunk) < self.win_inc:
                chunk = torch.nn.functional.pad(chunk, (0, self.win_inc - len(chunk)))

            self.input_buffer = torch.roll(self.input_buffer, -self.win_inc)
            self.input_buffer[-self.win_inc:] = chunk
            frames.append(torch.fft.rfft(self.input_buffer * self.window, n=self.fft_len))
            self.seen_first_frame = True

        if not frames:
            return None
        return torch.stack(frames, dim=-1)


class StreamingISTFTNaive:
    """Matches the current test_wav_streaming_offline_model_* overlap-add behavior."""

    def __init__(self, win_len=512, win_inc=256, fft_len=512, device="cpu"):
        self.win_len = win_len
        self.win_inc = win_inc
        self.fft_len = fft_len
        self.device = device
        self.window = torch.hamming_window(win_len, device=device)
        self.output_buffer = torch.zeros(win_len, device=device)
        self.prev_samples = torch.zeros(win_len - win_inc, device=device)

    def reset(self):
        self.output_buffer.zero_()
        self.prev_samples.zero_()

    def process(self, spec_frames):
        output_samples = []
        for frame_idx in range(spec_frames.shape[-1]):
            spec = spec_frames[..., frame_idx]
            framed = torch.fft.irfft(spec, n=self.fft_len)[:self.win_len]
            framed = framed * self.window

            self.output_buffer[:self.win_len - self.win_inc] = self.prev_samples
            self.output_buffer[self.win_len - self.win_inc:] = 0
            self.output_buffer += framed

            output_samples.append(self.output_buffer[:self.win_inc].clone())
            self.prev_samples = self.output_buffer[self.win_inc:].clone()

        return torch.cat(output_samples, dim=0).cpu().numpy() if output_samples else None

    def flush(self):
        tail = self.prev_samples.clone()
        self.prev_samples.zero_()
        self.output_buffer.zero_()
        return tail.cpu().numpy()


class StreamingISTFTNormalized:
    """Streaming WOLA ISTFT with window-square normalization, matching torch.istft semantics."""

    def __init__(self, win_len=512, win_inc=256, fft_len=512, device="cpu", eps=1e-8):
        self.win_len = win_len
        self.win_inc = win_inc
        self.fft_len = fft_len
        self.device = device
        self.eps = eps
        self.window = torch.hamming_window(win_len, device=device)
        self.window_square = self.window ** 2
        self.audio_buffer = torch.zeros(win_len, device=device)
        self.norm_buffer = torch.zeros(win_len, device=device)

    def reset(self):
        self.audio_buffer.zero_()
        self.norm_buffer.zero_()

    def process(self, spec_frames):
        output_samples = []
        for frame_idx in range(spec_frames.shape[-1]):
            spec = spec_frames[..., frame_idx]
            frame = torch.fft.irfft(spec, n=self.fft_len)[:self.win_len]

            self.audio_buffer += frame * self.window
            self.norm_buffer += self.window_square

            out = self.audio_buffer[:self.win_inc].clone()
            norm = torch.clamp(self.norm_buffer[:self.win_inc].clone(), min=self.eps)
            output_samples.append(out / norm)

            self.audio_buffer = torch.roll(self.audio_buffer, -self.win_inc)
            self.norm_buffer = torch.roll(self.norm_buffer, -self.win_inc)
            self.audio_buffer[-self.win_inc:] = 0
            self.norm_buffer[-self.win_inc:] = 0

        return torch.cat(output_samples, dim=0).cpu().numpy() if output_samples else None

    def flush(self):
        tail = self.audio_buffer[:self.win_len - self.win_inc].clone()
        norm = torch.clamp(self.norm_buffer[:self.win_len - self.win_inc].clone(), min=self.eps)
        self.reset()
        return (tail / norm).cpu().numpy()


def build_istft(mode, win_len, win_inc, fft_len, device):
    if mode == "naive":
        return StreamingISTFTNaive(win_len, win_inc, fft_len, device)
    if mode == "normalized":
        return StreamingISTFTNormalized(win_len, win_inc, fft_len, device)
    raise ValueError(f"Unsupported istft mode: {mode}")


def reconstruct_streaming(audio, win_len, win_inc, fft_len, buffer_frames, mode, device):
    stft = StreamingSTFT(win_len, win_inc, fft_len, device)
    istft = build_istft(mode, win_len, win_inc, fft_len, device)
    stft.reset()
    istft.reset()

    buffer_size = win_inc * buffer_frames
    outputs = []
    for start in range(0, len(audio), buffer_size):
        chunk = audio[start:start + buffer_size]
        spec = stft.process(chunk)
        if spec is None:
            continue
        wav_chunk = istft.process(spec)
        if wav_chunk is not None:
            outputs.append(wav_chunk)

    tail = istft.flush()
    if tail is not None and len(tail) > 0:
        outputs.append(tail)

    if not outputs:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(outputs).astype(np.float32)


def collect_mixture_paths(wav_dir, num_samples=None):
    paths = []
    for item in sorted(os.listdir(wav_dir)):
        sample_dir = os.path.join(wav_dir, item)
        mixture_path = os.path.join(sample_dir, "mixture.wav")
        if os.path.isdir(sample_dir) and os.path.exists(mixture_path):
            paths.append((item, mixture_path))
    if num_samples is not None:
        paths = paths[:num_samples]
    return paths


def process_one(sample_name, mixture_path, output_dir, args, device):
    fs, mixture = wav_read_float(mixture_path)
    reconstructed = reconstruct_streaming(
        mixture,
        args.win_len,
        args.win_inc,
        args.fft_len,
        args.buffer_frames,
        args.istft_mode,
        device,
    )
    reconstructed = reconstructed[:len(mixture)]

    sample_output_dir = os.path.join(output_dir, sample_name)
    output_path = os.path.join(sample_output_dir, f"mixture_reconstructed_{args.istft_mode}.wav")
    wav_write_float(output_path, fs, reconstructed)

    stats = error_stats(reconstructed, mixture)
    return output_path, stats


def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct mixture.wav with streaming STFT -> streaming ISTFT for diagnosis."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--wav_dir", type=str, help="Directory containing sample*/mixture.wav files")
    input_group.add_argument("--input_wav", type=str, help="Single wav file to reconstruct")
    parser.add_argument("--output_dir", type=str, default="./streaming_stft_istft_reconstruction")
    parser.add_argument("--istft_mode", choices=["naive", "normalized"], default="normalized")
    parser.add_argument("--win_len", type=int, default=512)
    parser.add_argument("--win_inc", type=int, default=256)
    parser.add_argument("--fft_len", type=int, default=512)
    parser.add_argument("--buffer_frames", type=int, default=100)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = parser.parse_args()

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but is not available")
        device = torch.device("cuda")
    elif args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    os.makedirs(args.output_dir, exist_ok=True)

    if args.input_wav is not None:
        items = [(os.path.splitext(os.path.basename(args.input_wav))[0], args.input_wav)]
    else:
        items = collect_mixture_paths(args.wav_dir, args.num_samples)

    print(f"Using device: {device}")
    print(f"ISTFT mode: {args.istft_mode}")
    print(f"Found {len(items)} mixture wavs")

    all_sisdr = []
    all_rms = []
    all_max_abs = []
    for sample_name, mixture_path in tqdm(items, desc="Reconstructing"):
        output_path, stats = process_one(sample_name, mixture_path, args.output_dir, args, device)
        all_sisdr.append(stats["sisdr"])
        all_rms.append(stats["rms_error"])
        all_max_abs.append(stats["max_abs_error"])
        print(
            f"{sample_name}: SI-SDR={stats['sisdr']:.2f} dB, "
            f"RMS err={stats['rms_error']:.6g}, "
            f"Max abs err={stats['max_abs_error']:.6g}, "
            f"out={output_path}"
        )

    if all_sisdr:
        print("\nSummary:")
        print(f"SI-SDR:      {np.mean(all_sisdr):.2f} +/- {np.std(all_sisdr):.2f} dB")
        print(f"RMS error:   {np.mean(all_rms):.6g} +/- {np.std(all_rms):.6g}")
        print(f"Max abs err: {np.mean(all_max_abs):.6g} +/- {np.std(all_max_abs):.6g}")


if __name__ == "__main__":
    main()
