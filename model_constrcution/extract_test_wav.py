
#!/usr/bin/env python3
import os
import sys
import h5py
import numpy as np
import scipy.io.wavfile as wav
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def Inverse_STFT(inputs, win_len=512, win_hop=256, fft_len=512):
    import torch
    import torch.nn.functional as F
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


def main():
    test_h5 = "./small_dataset/test_small.h5"
    output_dir = "./test_wav"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading test data from: {test_h5}")
    
    with h5py.File(test_h5, 'r') as f:
        X1 = torch.from_numpy(f['X1'][0]).float().unsqueeze(0)
        R1 = torch.from_numpy(f['R1'][0]).float().unsqueeze(0)
        R2 = torch.from_numpy(f['R2'][0]).float().unsqueeze(0)
        R3 = torch.from_numpy(f['R3'][0]).float().unsqueeze(0)
    
    print(f"X1 shape: {X1.shape}")
    print(f"R1 shape: {R1.shape}")
    
    mixture = Inverse_STFT(X1)
    
    fs = 16000
    
    mixture_np = mixture.squeeze().numpy()
    wav.write(os.path.join(output_dir, 'mixture.wav'), fs, 
              (mixture_np * 32767).astype(np.int16))
    
    wav.write(os.path.join(output_dir, 'speech_gt.wav'), fs, 
              (R1.squeeze().numpy() * 32767).astype(np.int16))
    
    wav.write(os.path.join(output_dir, 'music_gt.wav'), fs, 
              (R2.squeeze().numpy() * 32767).astype(np.int16))
    
    wav.write(os.path.join(output_dir, 'others_gt.wav'), fs, 
              (R3.squeeze().numpy() * 32767).astype(np.int16))
    
    print(f"Test wav files saved to: {output_dir}")
    print(f"  - mixture.wav")
    print(f"  - speech_gt.wav")
    print(f"  - music_gt.wav")
    print(f"  - others_gt.wav")


if __name__ == "__main__":
    main()
