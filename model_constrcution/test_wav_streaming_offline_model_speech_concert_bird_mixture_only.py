#!/usr/bin/env python3
import os
import sys
import argparse
import torch
import numpy as np
import scipy.io.wavfile as wav
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from DNN_models.Complex_MTASS_model import ComplexMTASSLightning
from DNN_models.Complex_MTASS import Complex_MTASS
from DNN_models.Complex_MTASS_Solver import Complex_MTASS_model


def load_separator(ckpt_path, device, chunk_size, chunk_frames, istft_mode):
    print("Loading model...")
    model = ComplexMTASSLightning.load_from_checkpoint(
        ckpt_path,
        map_location=device,
        model_class=Complex_MTASS,
        loss_class=Complex_MTASS_model,
    )
    model.to(device)
    model.eval()
    model.freeze()
    print("Model loaded!")

    return RealTimeAudioSeparator(
        model,
        win_len=512,
        win_inc=256,
        fft_len=512,
        chunk_size=chunk_size,
        chunk_frames=chunk_frames,
        istft_mode=istft_mode,
        device=device
    )


class StreamingSTFT:
    def __init__(self, win_len=512, win_inc=256, fft_len=512, device='cpu'):
        self.win_len = win_len
        self.win_inc = win_inc
        self.fft_len = fft_len
        self.device = device

        self.window = torch.hamming_window(win_len, device=device)
        self.input_buffer = torch.zeros(win_len, device=device)

    def reset(self):
        self.input_buffer = torch.zeros(self.win_len, device=self.device)

    def process(self, new_samples):
        new_samples = torch.from_numpy(new_samples).float().to(self.device)
        num_new = len(new_samples)

        frames = []

        if self.input_buffer.nonzero().numel() == 0 and num_new >= self.win_len:
            first_win = new_samples[:self.win_len]
            self.input_buffer = first_win
            framed = self.input_buffer * self.window
            spec = torch.fft.rfft(framed, n=self.fft_len)
            frames.append(spec)
            remaining_start = self.win_len
        else:
            remaining_start = 0

        for i in range(remaining_start, num_new, self.win_inc):
            chunk_end = min(i + self.win_inc, num_new)
            chunk = new_samples[i:chunk_end]

            self.input_buffer = torch.roll(self.input_buffer, -len(chunk))
            self.input_buffer[-len(chunk):] = chunk

            framed = self.input_buffer * self.window
            spec = torch.fft.rfft(framed, n=self.fft_len)
            frames.append(spec)

        if frames:
            return torch.stack(frames, dim=-1)
        return None


class StreamingISTFT:
    def __init__(self, win_len=512, win_inc=256, fft_len=512, device='cpu',
                 mode='naive', eps=1e-8):
        if mode not in ('naive', 'normalized'):
            raise ValueError(f"Unsupported ISTFT mode: {mode}")
        self.win_len = win_len
        self.win_inc = win_inc
        self.fft_len = fft_len
        self.device = device
        self.mode = mode
        self.eps = eps

        self.window = torch.hamming_window(win_len, device=device)
        self.output_buffer = torch.zeros(win_len, device=device)
        self.prev_samples = torch.zeros(win_len - win_inc, device=device)
        self.audio_buffer = torch.zeros(win_len, device=device)
        self.norm_buffer = torch.zeros(win_len, device=device)
        self.window_square = self.window ** 2

    def reset(self):
        self.output_buffer = torch.zeros(self.win_len, device=self.device)
        self.prev_samples = torch.zeros(self.win_len - self.win_inc, device=self.device)
        self.audio_buffer = torch.zeros(self.win_len, device=self.device)
        self.norm_buffer = torch.zeros(self.win_len, device=self.device)

    def process(self, spec_frames):
        if self.mode == 'normalized':
            return self._process_normalized(spec_frames)

        num_frames = spec_frames.shape[-1]

        output_samples = []

        for i in range(num_frames):
            spec = spec_frames[..., i]
            framed = torch.fft.irfft(spec, n=self.fft_len)
            framed = framed * self.window

            self.output_buffer[:self.win_len - self.win_inc] = self.prev_samples
            self.output_buffer[self.win_len - self.win_inc:] = 0
            self.output_buffer += framed

            output_samples.append(self.output_buffer[:self.win_inc].clone())
            self.prev_samples = self.output_buffer[self.win_inc:].clone()

        if output_samples:
            return torch.cat(output_samples, dim=0).cpu().numpy()
        return None

    def _process_normalized(self, spec_frames):
        output_samples = []

        for i in range(spec_frames.shape[-1]):
            spec = spec_frames[..., i]
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

        if output_samples:
            return torch.cat(output_samples, dim=0).cpu().numpy()
        return None

    def flush(self):
        if self.mode == 'normalized':
            tail = self.audio_buffer[:self.win_len - self.win_inc].clone()
            norm = torch.clamp(
                self.norm_buffer[:self.win_len - self.win_inc].clone(),
                min=self.eps,
            )
            self.audio_buffer.zero_()
            self.norm_buffer.zero_()
            return (tail / norm).cpu().numpy()

        if self.prev_samples.numel() == 0:
            return None

        tail = self.prev_samples.clone()
        self.prev_samples.zero_()
        self.output_buffer.zero_()
        return tail.cpu().numpy()


class RealTimeAudioSeparator:
    def __init__(self, model, win_len=512, win_inc=256, fft_len=512,
                 chunk_size=32, chunk_frames=100, istft_mode='naive', device='cpu'):
        if chunk_frames <= 0:
            raise ValueError(f"chunk_frames must be positive, got {chunk_frames}")
        self.model = model
        self.model.eval()
        self.device = device

        self.win_len = win_len
        self.win_inc = win_inc
        self.fft_len = fft_len
        self.chunk_size = chunk_size
        self.chunk_frames = chunk_frames
        self.istft_mode = istft_mode

        self.stft = StreamingSTFT(win_len, win_inc, fft_len, device)
        self.istft_speech = StreamingISTFT(win_len, win_inc, fft_len, device, mode=istft_mode)
        self.istft_concert = StreamingISTFT(win_len, win_inc, fft_len, device, mode=istft_mode)
        self.istft_bird = StreamingISTFT(win_len, win_inc, fft_len, device, mode=istft_mode)

    def reset(self):
        self.stft.reset()
        self.istft_speech.reset()
        self.istft_concert.reset()
        self.istft_bird.reset()
        reset_fn = getattr(self.model, 'reset_streaming_state', None)
        if callable(reset_fn):
            reset_fn()

    def _process_spec_chunk(self, new_spec):
        with torch.no_grad():
            stream_fn = getattr(self.model, 'forward_streaming', None)
            if callable(stream_fn):
                z1, z2, z3 = stream_fn(new_spec.unsqueeze(0))
            else:
                z1, z2, z3 = self.model(new_spec.unsqueeze(0))

        out1 = z1.squeeze(0)
        out2 = z2.squeeze(0)
        out3 = z3.squeeze(0)

        return out1, out2, out3

    def _spec_to_ri(self, spec):
        real = spec.real
        imag = spec.imag
        return torch.cat([real, imag], dim=0)

    def _ri_to_spec(self, ri):
        cutoff = self.fft_len // 2 + 1
        real = ri[:cutoff]
        imag = ri[cutoff:]
        return torch.complex(real, imag)

    def process(self, audio_samples):
        spec_frames = self.stft.process(audio_samples)

        if spec_frames is None:
            return None, None, None

        ri_spec = self._spec_to_ri(spec_frames)

        z1_ri, z2_ri, z3_ri = self._process_spec_chunk(ri_spec)

        z1_spec = self._ri_to_spec(z1_ri)
        z2_spec = self._ri_to_spec(z2_ri)
        z3_spec = self._ri_to_spec(z3_ri)

        speech_samples = self.istft_speech.process(z1_spec)
        concert_samples = self.istft_concert.process(z2_spec)
        bird_samples = self.istft_bird.process(z3_spec)

        return speech_samples, concert_samples, bird_samples

    def process_file(self, input_path, output_dir):
        self.reset()

        fs_read, audio_data = wav.read(input_path)
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32) / np.iinfo(audio_data.dtype).max

        if len(audio_data.shape) > 1:
            audio_data = audio_data[:, 0]

        os.makedirs(output_dir, exist_ok=True)

        buffer_size = self.win_inc * self.chunk_frames
        speech_output = []
        concert_output = []
        bird_output = []

        for i in range(0, len(audio_data), buffer_size):
            chunk = audio_data[i:i+buffer_size]
            speech, concert, bird = self.process(chunk)

            if speech is not None:
                speech_output.append(speech)
            if concert is not None:
                concert_output.append(concert)
            if bird is not None:
                bird_output.append(bird)

        speech_tail = self.istft_speech.flush()
        concert_tail = self.istft_concert.flush()
        bird_tail = self.istft_bird.flush()

        if speech_tail is not None:
            speech_output.append(speech_tail)
        if concert_tail is not None:
            concert_output.append(concert_tail)
        if bird_tail is not None:
            bird_output.append(bird_tail)

        speech_out = None
        concert_out = None
        bird_out = None
        stem = os.path.splitext(os.path.basename(input_path))[0]

        if speech_output:
            speech_out = np.concatenate(speech_output)
            wav.write(os.path.join(output_dir, f'{stem}_speech_es.wav'), fs_read,
                     (speech_out * 32767).astype(np.int16))

        if concert_output:
            concert_out = np.concatenate(concert_output)
            wav.write(os.path.join(output_dir, f'{stem}_concert_es.wav'), fs_read,
                     (concert_out * 32767).astype(np.int16))

        if bird_output:
            bird_out = np.concatenate(bird_output)
            wav.write(os.path.join(output_dir, f'{stem}_bird_es.wav'), fs_read,
                     (bird_out * 32767).astype(np.int16))

        return speech_out, concert_out, bird_out


def collect_wav_files(wav_dir):
    wav_files = []
    for item in sorted(os.listdir(wav_dir)):
        item_path = os.path.join(wav_dir, item)
        if os.path.isdir(item_path):
            for filename in sorted(os.listdir(item_path)):
                file_path = os.path.join(item_path, filename)
                if os.path.isfile(file_path) and filename.lower().endswith('.wav'):
                    wav_files.append(file_path)
        elif os.path.isfile(item_path) and item.lower().endswith('.wav'):
            wav_files.append(item_path)
    return wav_files


def get_output_dir(input_path, wav_dir, output_dir):
    parent_dir = os.path.dirname(input_path)
    rel_parent = os.path.relpath(parent_dir, wav_dir)
    if rel_parent == '.':
        return output_dir
    return os.path.join(output_dir, rel_parent)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wav_dir', type=str, required=True, help='Directory with test wav files')
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to Complex_MTASS checkpoint .ckpt')
    parser.add_argument('--output_dir', type=str, default='./wav_streaming_offline_model_results_speech_concert_bird_mixture_only', help='Folder to save results')
    parser.add_argument('--use_cuda', action='store_true', default=True)
    parser.add_argument('--chunk_size', type=int, default=32, help='Chunk size (frames)')
    parser.add_argument(
        '--chunk_frames',
        type=int,
        default=100,
        help='Number of STFT hop frames to process per streaming pipeline call',
    )
    parser.add_argument(
        '--istft_mode',
        type=str,
        default='naive',
        choices=['naive', 'normalized'],
        help='Streaming ISTFT mode. naive preserves the previous behavior; normalized applies window-square compensation.',
    )
    parser.add_argument('--num_samples', type=int, default=None, help='Number of wav files to process')

    args = parser.parse_args()

    device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Mode: inference_only")
    print(f"Chunk frames: {args.chunk_frames}")
    print(f"ISTFT mode: {args.istft_mode}")

    wav_files = collect_wav_files(args.wav_dir)

    if args.num_samples is not None:
        wav_files = wav_files[:args.num_samples]

    print(f"Found {len(wav_files)} wav files")

    os.makedirs(args.output_dir, exist_ok=True)

    separator = load_separator(
        args.ckpt_path,
        device,
        args.chunk_size,
        args.chunk_frames,
        args.istft_mode,
    )

    processed = 0
    for wav_path in tqdm(wav_files, desc="Processing wav files"):
        output_sample_dir = get_output_dir(wav_path, args.wav_dir, args.output_dir)
        separator.process_file(wav_path, output_sample_dir)
        processed += 1

    print("\n" + "=" * 60)
    print("Inference finished (Streaming with Complex_MTASS Offline Model)")
    print("=" * 60)
    print(f"Processed wav files: {processed}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
