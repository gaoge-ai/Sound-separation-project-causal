import os
import argparse
import torch
import numpy as np
import scipy.io.wavfile as wav
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from DNN_models.Complex_MTASS_model import ComplexMTASSLightning
from DNN_models.Complex_MTASS import *
from DNN_models.Complex_MTASS_Solver import *

def Inverse_STFT(inputs, win_len, win_hop, fft_len):
    cutoff = fft_len // 2 + 1
    real_part = inputs[:, :cutoff, :]
    imag_part = inputs[:, cutoff:, :]

    complex_spec = torch.complex(real_part, imag_part)
    istft_window = torch.hamming_window(win_len, device=inputs.device)

    reconstruction = torch.istft(
        complex_spec,
        n_fft=fft_len,
        hop_length=win_hop,
        win_length=win_len,
        window=istft_window,
        center=False,
        normalized=False,
        onesided=True,
        return_complex=False
    )

    return reconstruction

def STFT(wav_data, win_len, win_hop, fft_len):
    window = torch.hamming_window(win_len, device=wav_data.device)
    complex_spec = torch.stft(
        wav_data,
        n_fft=fft_len,
        hop_length=win_hop,
        win_length=win_len,
        window=window,
        center=False,
        normalized=False,
        onesided=True,
        return_complex=False
    )
    real_part = complex_spec[:, :, 0, :]
    imag_part = complex_spec[:, :, 1, :]
    features = torch.cat([real_part, imag_part], dim=1)
    return features

def wav_write(data, path, filename, fs):
    full_path = os.path.join(path, filename)
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()

    wav.write(full_path, fs, data)

def get_existing_classes(input_sample_dir):
    gt_map = {
        'speech_gt.wav': 'speech',
        'music_gt.wav': 'music',
        'others_gt.wav': 'others'
    }
    existing_classes = []
    for gt_file, class_name in gt_map.items():
        if os.path.exists(os.path.join(input_sample_dir, gt_file)):
            existing_classes.append(class_name)
    return existing_classes

def copy_gt_files(input_sample_dir, output_sample_dir):
    gt_files = ['speech_gt.wav', 'music_gt.wav', 'others_gt.wav']
    copied = []
    for gt_file in gt_files:
        src_path = os.path.join(input_sample_dir, gt_file)
        if os.path.exists(src_path):
            import shutil
            dst_path = os.path.join(output_sample_dir, gt_file)
            shutil.copy2(src_path, dst_path)
            copied.append(gt_file)
    return copied

class WAVDirectoryDataset(Dataset):
    def __init__(self, root_dir, win_len=512, win_inc=256, fft_len=512):
        self.root_dir = root_dir
        self.win_len = win_len
        self.win_inc = win_inc
        self.fft_len = fft_len
        self.wav_files = []

        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if (file.endswith('.wav') or file.endswith('.WAV')) and 'mixture' in file:
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, root_dir)
                    self.wav_files.append((full_path, rel_path))

        self.wav_files.sort()
        self.length = len(self.wav_files)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        wav_path, rel_path = self.wav_files[idx]
        fs, audio_data = wav.read(wav_path)

        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32) / 32768.0

        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)

        audio_tensor = torch.from_numpy(audio_data).float()
        audio_tensor = audio_tensor.unsqueeze(0)

        X1 = STFT(audio_tensor, self.win_len, self.win_inc, self.fft_len)
        X1 = X1.squeeze(0)

        return X1, rel_path, idx

def test(args):
    device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
    win_len = 512
    win_inc = 256
    fft_len = 512
    fs = args.sample_rate
    print(f"Testing on: {device}")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")

    model = ComplexMTASSLightning.load_from_checkpoint(
        args.ckpt_path,
        model_class=Complex_MTASS,
        loss_class=Complex_MTASS_model,
    )
    model.to(device)
    model.eval()
    model.freeze()

    test_dataset = WAVDirectoryDataset(args.input_dir, win_len=win_len, win_inc=win_inc, fft_len=fft_len)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    print(f"Found {len(test_dataset)} audio files to process")
    print("Start inference...")

    processed_count = 0
    with torch.no_grad():
        for batch in tqdm(test_loader):
            X1, rel_path, idx = batch
            X1 = X1.to(device)
            rel_path = rel_path[0]

            Z1, Z2, Z3 = model(X1)
            mixture_wav = Inverse_STFT(X1, win_len, win_inc, fft_len)

            Z1_time = Inverse_STFT(Z1, win_len, win_inc, fft_len)
            Z2_time = Inverse_STFT(Z2, win_len, win_inc, fft_len)
            Z3_time = Inverse_STFT(Z3, win_len, win_inc, fft_len)

            input_sample_dir = os.path.join(args.input_dir, os.path.dirname(rel_path))
            existing_classes = get_existing_classes(input_sample_dir)

            original_dirname = os.path.dirname(rel_path)
            if existing_classes:
                classes_suffix = '-'.join(existing_classes)
                new_dirname = f"{original_dirname}_{classes_suffix}"
            else:
                new_dirname = original_dirname

            output_path = os.path.join(args.output_dir, new_dirname)
            os.makedirs(output_path, exist_ok=True)

            if args.num_sources >= 1 and 'speech' in existing_classes:
                wav_write(Z1_time.squeeze(), output_path, "speech_es.wav", fs)
            if args.num_sources >= 2 and 'music' in existing_classes:
                wav_write(Z2_time.squeeze(), output_path, "music_es.wav", fs)
            if args.num_sources >= 3 and 'others' in existing_classes:
                wav_write(Z3_time.squeeze(), output_path, "others_es.wav", fs)

            if args.save_mixture:
                wav_write(mixture_wav.squeeze(), output_path, "mixture.wav", fs)

            if args.copy_gt:
                copy_gt_files(input_sample_dir, output_path)

            processed_count += 1

    print(f"\nProcessing completed! {processed_count} files processed.")
    print(f"Results saved to: {args.output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory containing wav files (will be searched recursively)')
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to model checkpoint .ckpt')
    parser.add_argument('--output_dir', type=str, default='./separated_results', help='Output directory to save separated wavs')
    parser.add_argument('--sample_rate', type=int, default=16000, help='Sample rate of input audio')
    parser.add_argument('--num_sources', type=int, choices=[2, 3, 4, 5], required=True,
                       help='混合声源数量: 2, 3, 4 或 5')
    parser.add_argument('--use_cuda', action='store_true', default=True)
    parser.add_argument('--save_mixture', action='store_true', default=False, help='Save original mixture wav')
    parser.add_argument('--copy_gt', action='store_true', default=True, help='Copy ground truth *_gt.wav files to output directory')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    test(args)
