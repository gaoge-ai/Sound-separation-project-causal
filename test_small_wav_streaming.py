
#!/usr/bin/env python3
import os
import sys

project_root = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(project_root, 'model_constrcution')
os.chdir(model_dir)
sys.path.insert(0, model_dir)

print("=" * 60)
print("Small Dataset WAV Chunk Streaming Test - SI-SDR")
print("=" * 60)

import test_wav_streaming_sdr as tws

import argparse
args = argparse.Namespace(
    wav_dir="./test_2mix_wavs",
    ckpt_path="experiments/small_test_streaming_cpu/checkpoints/epoch=0-step=2.ckpt",
    output_dir="./wav_streaming_results_small",
    use_cuda=False,
    history_size=256,
    chunk_size=32,
    num_samples=5
)

device = tws.torch.device("cuda" if args.use_cuda and tws.torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("\nLoading model...")
model = tws.ComplexMTASSLightningStreaming.load_from_checkpoint(
    args.ckpt_path,
    model_class=tws.Complex_MTASS_Streaming,
    loss_class=tws.Complex_MTASS_model,
)
model.to(device)
model.eval()
model.freeze()
print("Model loaded!")

separator = tws.RealTimeAudioSeparator(
    model,
    win_len=512,
    win_inc=256,
    fft_len=512,
    history_size=args.history_size,
    chunk_size=args.chunk_size,
    device=device
)

sample_dirs = []
for item in sorted(os.listdir(args.wav_dir)):
    item_path = os.path.join(args.wav_dir, item)
    if os.path.isdir(item_path):
        mixture_path = os.path.join(item_path, 'mixture.wav')
        if os.path.exists(mixture_path):
            sample_dirs.append(item_path)

if args.num_samples is not None:
    sample_dirs = sample_dirs[:args.num_samples]

print(f"Found {len(sample_dirs)} samples")

os.makedirs(args.output_dir, exist_ok=True)

all_speech_sdr = []
all_music_sdr = []
all_others_sdr = []
all_speech_sisdr = []
all_music_sisdr = []
all_others_sisdr = []

import tqdm
for sample_idx, sample_dir in enumerate(tqdm.tqdm(sample_dirs, desc="Processing samples")):
    sample_name = os.path.basename(sample_dir)
    output_sample_dir = os.path.join(args.output_dir, sample_name)
    
    mixture_path = os.path.join(sample_dir, 'mixture.wav')
    speech_gt_path = os.path.join(sample_dir, 'speech_gt.wav')
    music_gt_path = os.path.join(sample_dir, 'music_gt.wav')
    others_gt_path = os.path.join(sample_dir, 'others_gt.wav')
    
    fs, speech_gt = tws.wav.read(speech_gt_path)
    fs, music_gt = tws.wav.read(music_gt_path)
    fs, others_gt = tws.wav.read(others_gt_path)
    
    if speech_gt.dtype != np.float32:
        speech_gt = speech_gt.astype(np.float32) / 32767.0
        music_gt = music_gt.astype(np.float32) / 32767.0
        others_gt = others_gt.astype(np.float32) / 32767.0
    
    speech_es, music_es, others_es = separator.process_file(mixture_path, output_sample_dir, fs)
    
    if speech_es is not None:
        min_len = min(len(speech_es), len(speech_gt))
        speech_sdr = tws.sdr_cost(speech_es[:min_len], speech_gt[:min_len])
        speech_sisdr = tws.sisdr_cost(speech_es[:min_len], speech_gt[:min_len])
        all_speech_sdr.append(speech_sdr)
        all_speech_sisdr.append(speech_sisdr)
        print(f"  Sample {sample_idx} - Speech SI-SDR: {speech_sisdr:.2f}")
    
    if music_es is not None:
        min_len = min(len(music_es), len(music_gt))
        music_sdr = tws.sdr_cost(music_es[:min_len], music_gt[:min_len])
        music_sisdr = tws.sisdr_cost(music_es[:min_len], music_gt[:min_len])
        all_music_sdr.append(music_sdr)
        all_music_sisdr.append(music_sisdr)
        print(f"  Sample {sample_idx} - Music SI-SDR: {music_sisdr:.2f}")
    
    if others_es is not None:
        min_len = min(len(others_es), len(others_gt))
        others_sdr = tws.sdr_cost(others_es[:min_len], others_gt[:min_len])
        others_sisdr = tws.sisdr_cost(others_es[:min_len], others_gt[:min_len])
        all_others_sdr.append(others_sdr)
        all_others_sisdr.append(others_sisdr)
        print(f"  Sample {sample_idx} - Others SI-SDR: {others_sisdr:.2f}")

print("\n" + "="*60)
print("SI-SDR Statistics:")
print("="*60)

import numpy as np
if all_speech_sisdr:
    print(f"Speech SI-SDR: {np.mean(all_speech_sisdr):.2f} +/- {np.std(all_speech_sisdr):.2f}")

if all_music_sisdr:
    print(f"Music SI-SDR: {np.mean(all_music_sisdr):.2f} +/- {np.std(all_music_sisdr):.2f}")

if all_others_sisdr:
    print(f"Others SI-SDR: {np.mean(all_others_sisdr):.2f} +/- {np.std(all_others_sisdr):.2f}")

all_sisdr = all_speech_sisdr + all_music_sisdr + all_others_sisdr

if all_sisdr:
    print(f"\nTotal Average SI-SDR: {np.mean(all_sisdr):.2f} +/- {np.std(all_sisdr):.2f}")

print("="*60)
