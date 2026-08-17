import torch
import torch.nn.functional as F

def enframe(signal_batch, frame_size, frame_shift, window_gpu):
    batch_size, num_samples = signal_batch.shape
    if num_samples <= frame_size:
        nf = 1
    else:
        nf = (num_samples - frame_size + frame_shift - 1) // frame_shift + 1
    pad_length = (nf - 1) * frame_shift + frame_size
    padding_size = pad_length - num_samples
    if padding_size > 0:
        signal_batch = F.pad(signal_batch, (0, padding_size), mode='constant', value=0.0)
    frames = signal_batch.unsqueeze(1).unfold(2, frame_size, frame_shift)
    frames = frames.squeeze(1)
    frames = frames.transpose(1, 2)
    frames = frames * window_gpu.view(1, -1, 1)  
    return frames


def compute_fft(frames, frame_size):
    return torch.fft.rfft(frames, n=frame_size, dim=1)

def RI_split(complex_spec, n_fft=None):
    real = complex_spec.real
    imag = complex_spec.imag

    return torch.cat([real, imag], dim=1)

def overlap_add_batch(frames_np, inc):
    frames = torch.from_numpy(frames_np)
    frame_size, num_frames = frames.shape
    out_len = (num_frames - 1) * inc + frame_size
    sig = torch.zeros(out_len, dtype=torch.float32)
    for i in range(num_frames):
        start = i * inc
        end = start + frame_size
        sig[start:end] += frames[:, i]
        
    return sig.numpy()