import argparse
import os
import csv
import torch
import librosa
import torchaudio
import numpy as np
import h5py
import time
from tqdm import tqdm
from scipy import signal
from torch.utils.data import Dataset, DataLoader
from utils.utils_library_gpu import *

TARGET_SAMPLE_RATE = 16000
BATCH_SIZE = 32
DEVICE = 'cuda'
NUM_SOURCES = 3
FRAME_SIZE = 512
FRAME_SHIFT = 256
REJOIN_LEN = 624

class AudioDataset(Dataset):
    def __init__(self, csv_path_list, subset='train',
                 target_sample_rate=16000, target_duration=10):
        self.target_sample_rate = target_sample_rate
        self.target_duration = target_duration
        self.target_num_samples = self.target_sample_rate * self.target_duration
        self.src_names = []
        self.src_labels = []
        self.src_snrs = []
        for csv_file in csv_path_list:
            with open(csv_file, 'r', encoding='utf-8') as d:
                reader = csv.reader(d, skipinitialspace=True)
                head = next(reader)
                for row in reader:
                    names = []
                    labels = []
                    snrs = []
                    for i in range(0, len(row), 3):
                        names.append(row[i])
                        labels.append(row[i + 1])
                        snrs.append(row[i + 2])
                    self.src_names.append(names)
                    self.src_labels.append(labels)
                    self.src_snrs.append(snrs)

    def __len__(self):
        return len(self.src_names)

    def load_wav(self, path):
        max_length = self.target_sample_rate * 10
        wav = librosa.core.load(path, sr=self.target_sample_rate)[0]
        if len(wav) > max_length:
            wav = wav[0:max_length]
        # pad audio to max length, 10s for AudioCaps
        if len(wav) < max_length:
            # audio = torch.nn.functional.pad(audio, (0, self.max_length - audio.size(1)), 'constant')
            wav = np.pad(wav, (0, max_length - len(wav)), 'constant')
        return wav

    def mix_audios(self, audios, snrs):
        target = audios[0]
        target_energy = torch.sum(target ** 2)
        mixed = target.clone()
        scaled_audios = [target]
        for i in range(1, len(audios)):
            noise = audios[i]
            noise_energy = torch.sum(noise ** 2)
            snr_db = float(snrs[i])
            snr_linear = 10 ** (snr_db / 10)
            scale = torch.sqrt((target_energy / snr_linear) / (noise_energy + 1e-8))
            scaled_noise = noise * scale
            mixed += scaled_noise
            scaled_audios.append(scaled_noise)
        
        max_value = torch.max(torch.abs(mixed))
        if max_value > 1:
            mixed *= 0.9 / max_value
            scaled_audios = [audio * 0.9 / max_value for audio in scaled_audios]

        return mixed, scaled_audios

    def __getitem__(self, idx):
        src_name = self.src_names[idx]
        src_labels = self.src_labels[idx]
        src_snrs = self.src_snrs[idx]

        audios = [torch.from_numpy(self.load_wav(x)) for x in src_name]
        mixed_wav, scaled_sources = self.mix_audios(audios, src_snrs)

        s1 = torch.zeros_like(mixed_wav)
        s2 = torch.zeros_like(mixed_wav)
        s3 = torch.zeros_like(mixed_wav)

        for i, label in enumerate(src_labels):
            label_lower = label.lower()
            current_source = scaled_sources[i]

            if 'speech' in label_lower:
                s1 += current_source
            elif 'concert' in label_lower:
                s2 += current_source
            elif 'bird' in label_lower:
                s3 += current_source
            else:
                raise ValueError(f"未知类别标签: {label}")
        
        return mixed_wav, s1, s2, s3

def feature_extraction(input_csv_list, output_dir, split):
    os.makedirs(output_dir, exist_ok=True)
    h5_path = os.path.join(output_dir, f'{split}_ready.h5')
    dataset = AudioDataset(input_csv_list, target_sample_rate=TARGET_SAMPLE_RATE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=16, shuffle=False)
    frame_size = FRAME_SIZE
    frame_shift = int(frame_size / 2)
    winfunc = signal.windows.hamming(frame_size)
    winfunc_gpu = torch.tensor(winfunc, dtype=torch.float32).to(DEVICE)

    with h5py.File(h5_path, 'w') as h5f:
        h5f.attrs['dataset_name'] = split
        h5f.attrs['num_files'] = len(dataset)
        h5f.attrs['created_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
        
        h5f.create_dataset('X1', shape=(0, 514, REJOIN_LEN), maxshape=(None, 514, REJOIN_LEN), 
                          chunks=(32, 514, REJOIN_LEN), compression='gzip', dtype='float32')
        r_len = 256 * (624 - 1) + 512
        for i in range(1, NUM_SOURCES + 1):
            h5f.create_dataset(f'Y{i}', shape=(0, 514, REJOIN_LEN), maxshape=(None, 514, REJOIN_LEN), 
                               chunks=(32, 514, REJOIN_LEN), compression='gzip', dtype='float32')
            h5f.create_dataset(f'R{i}', shape=(0, r_len), maxshape=(None, r_len), 
                               chunks=(32, r_len), compression='gzip', dtype='float32')
    total_samples = 0
    for batch_idx, (mixture_wav, s1_wav, s2_wav, s3_wav) in enumerate(tqdm(dataloader, desc="Processing batches")):
        batch_size = mixture_wav.shape[0]
        sources_wav_list = [s1_wav, s2_wav, s3_wav]

        with torch.no_grad():
            mixture_wav = mixture_wav.to(DEVICE)
            mix_split = enframe(mixture_wav, frame_size, frame_shift, winfunc_gpu)
            mix_freq = compute_fft(mix_split, frame_size)
            # RI Split -> X1 [B, 514, T]
            mix_ri = RI_split(mix_freq, mix_freq.shape[1])

            sources_freq_ri_list = []
            sources_time_list = []
            for i in range(NUM_SOURCES):
                src_wav = sources_wav_list[i].to(DEVICE)
                src_split = enframe(src_wav, frame_size, frame_shift, winfunc_gpu)
                src_freq = compute_fft(src_split, frame_size)
                src_ri = RI_split(src_freq, src_freq.shape[1])
                sources_freq_ri_list.append(src_ri)
                sources_time_list.append(src_wav)

        mix_ri_np = mix_ri.cpu().numpy().astype(np.float32)
        
        with h5py.File(h5_path, 'a') as h5f:
            current_size = h5f['X1'].shape[0]
            new_size = current_size + batch_size
            
            # Save X1
            h5f['X1'].resize(new_size, axis=0)
            h5f['X1'][current_size:new_size] = mix_ri_np
            
            # Save Y1-Y3
            for i in range(NUM_SOURCES):
                y_np = sources_freq_ri_list[i].cpu().numpy().astype(np.float32)
                h5f[f'Y{i+1}'].resize(new_size, axis=0)
                h5f[f'Y{i+1}'][current_size:new_size] = y_np

            # Save R1-R3
            for i in range(NUM_SOURCES):
                t_frames_np = sources_time_list[i].cpu().numpy().astype(np.float32) # [B, T]
                h5f[f'R{i+1}'].resize(new_size, axis=0)
                h5f[f'R{i+1}'][current_size:new_size] = t_frames_np
        total_samples += batch_size
        print(f"Processed batch {batch_idx + 1}, Total samples: {total_samples}")
        torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process audio files")
    parser.add_argument('--input_csv_list', type=str, nargs='+', required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--split', type=str, default='train', help='Index of the split to process')
    args = parser.parse_args()

    feature_extraction(args.input_csv_list, args.output_dir, args.split)
