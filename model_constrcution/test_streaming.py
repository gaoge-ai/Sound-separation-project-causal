
import os
import argparse
import torch
import h5py
import numpy as np
import scipy.io.wavfile as wav
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from DNN_models.Complex_MTASS_model_streaming import ComplexMTASSLightningStreaming
from DNN_models.Complex_MTASS_streaming import *
from DNN_models.Complex_MTASS_Solver import *

def masked_metric(estimate, target, eps=1e-8):
    target_energy = torch.sum(target ** 2, dim=-1)
    mask = target_energy > eps
    batch_size = estimate.shape[0]
    sdr_vector = torch.zeros(batch_size, device=estimate.device)
    sisdr_vector = torch.zeros(batch_size, device=estimate.device)
    if mask.sum() > 0:
        valid_est = estimate[mask]
        valid_tgt = target[mask]

        valid_sdr = sdr_cost(valid_est, valid_tgt)
        sdr_vector[mask] = valid_sdr

        valid_sisdr = sisdr_cost(valid_est, valid_tgt)
        sisdr_vector[mask] = valid_sisdr
    mask_float = mask.float()
    return sdr_vector, sisdr_vector, mask_float
  
def compute_out_cost(mix, Z1, Z2, Z3, R1, R2, R3):
    win_len = 512
    win_inc = 256
    fft_len = 512
    Z1_time = Inverse_STFT(Z1, win_len, win_inc, fft_len)
    Z2_time = Inverse_STFT(Z2, win_len, win_inc, fft_len)
    Z3_time = Inverse_STFT(Z3, win_len, win_inc, fft_len)

    sdr_s, sisdr_s, mask_s = masked_metric(Z1_time, R1)
    sdr_m, sisdr_m, mask_m = masked_metric(Z2_time, R2)
    sdr_n, sisdr_n, mask_n = masked_metric(Z3_time, R3)
    total_mask = torch.tensor([mask_s.item(), mask_m.item(), mask_n.item()])

    sum_sdr = sdr_s + sdr_m + sdr_n 
    sum_sisdr = sisdr_s + sisdr_m + sisdr_n

    num_tasks = mask_s + mask_m + mask_n
    num_tasks = torch.clamp(num_tasks, min=1.0)
    
    per_sample_sdr = sum_sdr / num_tasks
    per_sample_sisdr = sum_sisdr / num_tasks

    total_sdr = torch.mean(per_sample_sdr)
    total_sisdr = torch.mean(per_sample_sisdr)

    speech_sisdr = torch.mean(sisdr_s)
    music_sisdr = torch.mean(sisdr_m)
    others_sisdr = torch.mean(sisdr_n)
    return total_sdr, total_sisdr, Z1_time, Z2_time, Z3_time, speech_sisdr, music_sisdr, others_sisdr, total_mask

def sdr_standard(estimated, target, eps=1e-8):
    signal_pow = torch.sum(target ** 2, dim=-1) + eps
    noise = estimated - target
    noise_pow = torch.sum(noise ** 2, dim=-1) + eps
    
    sdr = 10 * torch.log10(signal_pow / noise_pow)
    return sdr

def sdr_cost(estimated, target, eps=1e-8):
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
    return sdr

def sisdr_cost(estimated, target, eps=1e-8):
    dot = torch.sum(estimated * target, dim=-1, keepdim=True) + eps
    s_energy = torch.sum(target ** 2, dim=-1, keepdim=True) + eps

    scale = dot / s_energy
    target_scaled = scale * target
    e_noise = estimated - target_scaled
    target_pow = torch.sum(target_scaled ** 2, dim=-1) + eps
    noise_pow = torch.sum(e_noise ** 2, dim=-1) + eps
        
    sisdr = 10 * torch.log10(target_pow / noise_pow)
    return sisdr.squeeze(-1)

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

def wav_write(data, path, filename, fs):
    full_path = os.path.join(path, filename)
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    
    wav.write(full_path, fs, data)

def create_sisdr_string(total_mask, sisdr_values, labels, format_spec=".2f"):
    active_labels = [label for label, mask in zip(labels, total_mask.bool()) if mask]
    parts = []
    for label in active_labels:
        value = sisdr_values[label]
        parts.append(f"{label}{value:{format_spec}}")
    parts.append(f"{sisdr_values['total']:{format_spec}}")
    return "-".join(parts)

class HDF5Dataset(Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path
        self.h5_file = None
        with h5py.File(h5_path, 'r') as f:
            self.length = f['X1'].shape[0]
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')
            
        X1 = torch.from_numpy(self.h5_file['X1'][idx]).float()
        R_targets = [torch.from_numpy(self.h5_file[f'R{i}'][idx]).float() for i in range(1, 4)]
        
        return X1, R_targets, idx

def test(args):
    device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
    win_len = 512
    win_inc = 256
    fft_len = 512
    fs = 16000
    print(f"Testing on: {device}")
    model = ComplexMTASSLightningStreaming.load_from_checkpoint(
        args.ckpt_path,
        model_class=Complex_MTASS_Streaming,
        loss_class=Complex_MTASS_model,
    )
    model.to(device)
    model.eval()
    model.freeze()

    test_dataset = HDF5Dataset(args.test_h5)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=1,
        shuffle=False, 
        num_workers=4
    )

    print("Start inference...")
    labels = ["speech", "music", "others"]
    total_sdr_list = []
    total_sisdr_list = []

    with torch.no_grad():
        for batch in tqdm(test_loader):
            X1, R_targets, idx = batch
            X1 = X1.to(device)

            R_gt_speech = R_targets[0].to(device)
            R_gt_music  = R_targets[1].to(device)
            R_gt_others  = R_targets[2].to(device)

            Z1, Z2, Z3 = model(X1)
            mixture = Inverse_STFT(X1, win_len, win_inc, fft_len)
            total_sdr, total_sisdr, Z1_time, Z2_time, Z3_time, speech_sisdr, music_sisdr, others_sisdr, total_mask =  compute_out_cost(mixture, Z1, Z2, Z3, R_gt_speech, R_gt_music, R_gt_others)
            sisdr_values = {
                'total': total_sisdr,
                'speech': speech_sisdr,
                'music': music_sisdr,
                'others': others_sisdr
            }
            total_sdr_list.append(total_sdr)
            total_sisdr_list.append(total_sisdr)

            result_label = create_sisdr_string(total_mask, sisdr_values, labels, ".2f")
            save_dir = os.path.join(args.output_dir, f"sample{idx.item()}_{result_label}")
            os.makedirs(save_dir, exist_ok=True)
            wav_write(mixture.squeeze(), save_dir, "mixture.wav", fs)
            wav_write(R_gt_speech.squeeze(), save_dir, "speech_gt.wav", fs)
            wav_write(R_gt_music.squeeze(), save_dir, "music_gt.wav", fs)
            wav_write(R_gt_others.squeeze(), save_dir, "others_gt.wav", fs)
            wav_write(Z1_time.squeeze(), save_dir, "speech_es.wav", fs)
            wav_write(Z2_time.squeeze(),  save_dir,  "music_es.wav", fs)
            wav_write(Z3_time.squeeze(),  save_dir,  "others_es.wav", fs)

    avg_sdr = torch.mean(torch.tensor(total_sdr_list))
    avg_sisdr = torch.mean(torch.tensor(total_sisdr_list))

    print(f"Total SDR: {avg_sdr:.4f}")
    print(f"Total SI-SDR: {avg_sisdr:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_h5', type=str, required=True, help='Path to test data .h5')
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to model checkpoint .ckpt')
    parser.add_argument('--output_dir', type=str, default='./test_results_streaming', help='Folder to save wavs')
    parser.add_argument('--num_sources', type=int, choices=[2, 3, 4, 5], required=True,
                       help='混合声源数量: 2, 3, 4 或 5')
    parser.add_argument('--use_cuda', action='store_true', default=True)
    
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    test(args)
