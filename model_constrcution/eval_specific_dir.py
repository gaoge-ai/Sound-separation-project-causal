
#!/usr/bin/env python3
import os
import sys
import argparse
import torch
import numpy as np
import scipy.io.wavfile as wav
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, required=True, help='Directory with separated results')
    parser.add_argument('--num_samples', type=int, default=None, help='Number of samples to test')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Calculate SDR/SI-SDR for Specific Directory")
    print("=" * 60)
    print(f"Results dir: {args.results_dir}")
    print()
    
    sample_dirs = []
    for item in sorted(os.listdir(args.results_dir)):
        item_path = os.path.join(args.results_dir, item)
        if os.path.isdir(item_path):
            mixture_path = os.path.join(item_path, 'mixture.wav')
            if os.path.exists(mixture_path):
                sample_dirs.append(item_path)
    
    if args.num_samples is not None:
        sample_dirs = sample_dirs[:args.num_samples]
    
    print(f"Found {len(sample_dirs)} samples")
    print()
    
    all_speech_sdr = []
    all_music_sdr = []
    all_others_sdr = []
    all_speech_sisdr = []
    all_music_sisdr = []
    all_others_sisdr = []
    
    eps = 1e-8
    
    print("=" * 60)
    print("Calculating metrics...")
    print("=" * 60)
    
    for sample_idx, sample_dir in enumerate(tqdm(sample_dirs, desc="Processing samples")):
        sample_name = os.path.basename(sample_dir)
        
        speech_gt_path = os.path.join(sample_dir, 'speech_gt.wav')
        music_gt_path = os.path.join(sample_dir, 'music_gt.wav')
        others_gt_path = os.path.join(sample_dir, 'others_gt.wav')
        
        speech_es_path = os.path.join(sample_dir, 'speech_es.wav')
        music_es_path = os.path.join(sample_dir, 'music_es.wav')
        others_es_path = os.path.join(sample_dir, 'others_es.wav')
        
        fs, speech_gt = wav.read(speech_gt_path)
        fs, music_gt = wav.read(music_gt_path)
        fs, others_gt = wav.read(others_gt_path)
        
        if speech_gt.dtype != np.float32:
            speech_gt = speech_gt.astype(np.float32) / 32767.0
            music_gt = music_gt.astype(np.float32) / 32767.0
            others_gt = others_gt.astype(np.float32) / 32767.0
        
        valid_speech = np.sum(speech_gt ** 2) > eps
        valid_music = np.sum(music_gt ** 2) > eps
        valid_others = np.sum(others_gt ** 2) > eps
        
        print(f"\nSample {sample_idx}: {sample_name}")
        print(f"  Valid speech: {valid_speech} (energy: {np.sum(speech_gt ** 2):.6e})")
        print(f"  Valid music: {valid_music} (energy: {np.sum(music_gt ** 2):.6e})")
        print(f"  Valid others: {valid_others} (energy: {np.sum(others_gt ** 2):.6e})")
        
        speech_es = None
        music_es = None
        others_es = None
        
        if os.path.exists(speech_es_path):
            fs, speech_es = wav.read(speech_es_path)
            if speech_es.dtype != np.float32:
                speech_es = speech_es.astype(np.float32) / 32767.0
        
        if os.path.exists(music_es_path):
            fs, music_es = wav.read(music_es_path)
            if music_es.dtype != np.float32:
                music_es = music_es.astype(np.float32) / 32767.0
        
        if os.path.exists(others_es_path):
            fs, others_es = wav.read(others_es_path)
            if others_es.dtype != np.float32:
                others_es = others_es.astype(np.float32) / 32767.0
        
        if valid_speech and speech_es is not None:
            min_len = min(len(speech_es), len(speech_gt))
            speech_sdr = sdr_cost(speech_es[:min_len], speech_gt[:min_len))
            speech_sisdr = sisdr_cost(speech_es[:min_len], speech_gt[:min_len))
            all_speech_sdr.append(speech_sdr)
            all_speech_sisdr.append(speech_sisdr)
            print(f"  Speech - SDR: {speech_sdr:.2f}, SI-SDR: {speech_sisdr:.2f}")
        
        if valid_music and music_es is not None:
            min_len = min(len(music_es), len(music_gt))
            music_sdr = sdr_cost(music_es[:min_len], music_gt[:min_len))
            music_sisdr = sisdr_cost(music_es[:min_len], music_gt[:min_len))
            all_music_sdr.append(music_sdr)
            all_music_sisdr.append(music_sisdr)
            print(f"  Music - SDR: {music_sdr:.2f}, SI-SDR: {music_sisdr:.2f}")
        
        if valid_others and others_es is not None:
            min_len = min(len(others_es), len(others_gt))
            others_sdr = sdr_cost(others_es[:min_len], others_gt[:min_len))
            others_sisdr = sisdr_cost(others_es[:min_len], others_gt[:min_len))
            all_others_sdr.append(others_sdr)
            all_others_sisdr.append(others_sisdr)
            print(f"  Others - SDR: {others_sdr:.2f}, SI-SDR: {others_sisdr:.2f}")
    
    print("\n" + "="*60)
    print("Final Statistics (Valid Classes Only):")
    print("="*60)
    
    if all_speech_sdr:
        print(f"Speech SDR: {np.mean(all_speech_sdr):.2f} +/- {np.std(all_speech_sdr):.2f}")
        print(f"Speech SI-SDR: {np.mean(all_speech_sisdr):.2f} +/- {np.std(all_speech_sisdr):.2f}")
    
    if all_music_sdr:
        print(f"Music SDR: {np.mean(all_music_sdr):.2f} +/- {np.std(all_music_sdr):.2f}")
        print(f"Music SI-SDR: {np.mean(all_music_sisdr):.2f} +/- {np.std(all_music_sisdr):.2f}")
    
    if all_others_sdr:
        print(f"Others SDR: {np.mean(all_others_sdr):.2f} +/- {np.std(all_others_sdr):.2f}")
        print(f"Others SI-SDR: {np.mean(all_others_sisdr):.2f} +/- {np.std(all_others_sisdr):.2f}")
    
    all_sdr = all_speech_sdr + all_music_sdr + all_others_sdr
    all_sisdr = all_speech_sisdr + all_music_sisdr + all_others_sisdr
    
    if all_sdr:
        print(f"\nTotal Average SDR: {np.mean(all_sdr):.2f} +/- {np.std(all_sdr):.2f}")
        print(f"Total Average SI-SDR: {np.mean(all_sisdr):.2f} +/- {np.std(all_sisdr):.2f}")
    
    print("="*60)


if __name__ == '__main__':
    main()
