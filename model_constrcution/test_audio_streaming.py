
#!/usr/bin/env python3
import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import scipy.io.wavfile as wav
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from DNN_models.Complex_MTASS_model_streaming import ComplexMTASSLightningStreaming
from DNN_models.Complex_MTASS_streaming import Complex_MTASS_Streaming
from DNN_models.Complex_MTASS_Solver import Complex_MTASS_model


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
        new_samples = new_samples.to(self.device)
        num_new = len(new_samples)
        
        frames = []
        
        for i in range(0, num_new, self.win_inc):
            chunk_end = min(i + self.win_inc, num_new)
            chunk = new_samples[i:chunk_end]
            
            self.input_buffer = torch.roll(self.input_buffer, -len(chunk))
            self.input_buffer[-len(chunk):] = chunk
            
            if i + self.win_inc <= num_new or i == 0:
                framed = self.input_buffer * self.window
                spec = torch.fft.rfft(framed, n=self.fft_len)
                frames.append(spec)
        
        if frames:
            return torch.stack(frames, dim=-1)
        return None


class StreamingISTFT:
    def __init__(self, win_len=512, win_inc=256, fft_len=512, device='cpu'):
        self.win_len = win_len
        self.win_inc = win_inc
        self.fft_len = fft_len
        self.device = device
        
        self.window = torch.hamming_window(win_len, device=device)
        self.output_buffer = torch.zeros(win_len, device=device)
        self.prev_samples = torch.zeros(win_len - win_inc, device=device)
        
    def reset(self):
        self.output_buffer = torch.zeros(self.win_len, device=self.device)
        self.prev_samples = torch.zeros(self.win_len - self.win_inc, device=self.device)
    
    def process(self, spec_frames):
        spec_frames = spec_frames.to(self.device)
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
            return torch.cat(output_samples, dim=0)
        return None


class RealTimeAudioSeparator:
    def __init__(self, model, win_len=512, win_inc=256, fft_len=512, 
                 history_size=256, chunk_size=32, device='cpu'):
        self.model = model
        self.model.eval()
        self.device = device
        
        self.win_len = win_len
        self.win_inc = win_inc
        self.fft_len = fft_len
        self.history_size = history_size
        self.chunk_size = chunk_size
        
        self.stft = StreamingSTFT(win_len, win_inc, fft_len, device)
        self.istft_speech = StreamingISTFT(win_len, win_inc, fft_len, device)
        self.istft_music = StreamingISTFT(win_len, win_inc, fft_len, device)
        self.istft_others = StreamingISTFT(win_len, win_inc, fft_len, device)
        
        self.spec_buffer = None
        
    def reset(self):
        self.stft.reset()
        self.istft_speech.reset()
        self.istft_music.reset()
        self.istft_others.reset()
        self.spec_buffer = None
    
    def _init_spec_buffer(self, first_spec):
        pad_size = self.history_size - first_spec.shape[-1]
        if pad_size > 0:
            padded = F.pad(first_spec, (pad_size, 0))
            self.spec_buffer = padded
        else:
            self.spec_buffer = first_spec[..., -self.history_size:]
    
    def _process_spec_chunk(self, new_spec):
        if self.spec_buffer is None:
            self._init_spec_buffer(new_spec)
        else:
            self.spec_buffer = torch.cat(
                [self.spec_buffer, new_spec], dim=-1
            )[..., -self.history_size:]
        
        with torch.no_grad():
            z1, z2, z3 = self.model(self.spec_buffer.unsqueeze(0))
        
        out1 = z1[..., -new_spec.shape[-1]:].squeeze(0)
        out2 = z2[..., -new_spec.shape[-1]:].squeeze(0)
        out3 = z3[..., -new_spec.shape[-1]:].squeeze(0)
        
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
        audio_samples = torch.from_numpy(audio_samples).float().to(self.device)
        
        spec_frames = self.stft.process(audio_samples)
        
        if spec_frames is None:
            return None, None, None
        
        ri_spec = self._spec_to_ri(spec_frames)
        
        z1_ri, z2_ri, z3_ri = self._process_spec_chunk(ri_spec)
        
        z1_spec = self._ri_to_spec(z1_ri)
        z2_spec = self._ri_to_spec(z2_ri)
        z3_spec = self._ri_to_spec(z3_ri)
        
        speech_samples = self.istft_speech.process(z1_spec)
        music_samples = self.istft_music.process(z2_spec)
        others_samples = self.istft_others.process(z3_spec)
        
        return speech_samples, music_samples, others_samples
    
    def process_file(self, input_path, output_dir, fs=16000):
        self.reset()
        
        fs_read, audio_data = wav.read(input_path)
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32) / np.iinfo(audio_data.dtype).max
        
        if len(audio_data.shape) > 1:
            audio_data = audio_data[:, 0]
        
        os.makedirs(output_dir, exist_ok=True)
        
        buffer_size = self.win_inc * 100
        speech_output = []
        music_output = []
        others_output = []
        
        print(f"Processing audio file: {input_path}")
        print(f"Audio length: {len(audio_data)/fs:.2f} seconds")
        
        for i in tqdm(range(0, len(audio_data), buffer_size)):
            chunk = audio_data[i:i+buffer_size]
            speech, music, others = self.process(chunk)
            
            if speech is not None:
                speech_output.append(speech.cpu().numpy())
            if music is not None:
                music_output.append(music.cpu().numpy())
            if others is not None:
                others_output.append(others.cpu().numpy())
        
        if speech_output:
            speech_out = np.concatenate(speech_output)
            wav.write(os.path.join(output_dir, 'speech.wav'), fs, 
                     (speech_out * 32767).astype(np.int16))
        
        if music_output:
            music_out = np.concatenate(music_output)
            wav.write(os.path.join(output_dir, 'music.wav'), fs, 
                     (music_out * 32767).astype(np.int16))
        
        if others_output:
            others_out = np.concatenate(others_output)
            wav.write(os.path.join(output_dir, 'others.wav'), fs, 
                     (others_out * 32767).astype(np.int16))
        
        wav.write(os.path.join(output_dir, 'mixture.wav'), fs, 
                 (audio_data * 32767).astype(np.int16))
        
        print(f"Output saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_wav', type=str, required=True, help='Path to input wav file')
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to model checkpoint .ckpt')
    parser.add_argument('--output_dir', type=str, default='./audio_streaming_results', help='Folder to save separated wavs')
    parser.add_argument('--use_cuda', action='store_true', default=True)
    parser.add_argument('--history_size', type=int, default=256, help='History buffer size (frames)')
    parser.add_argument('--chunk_size', type=int, default=32, help='Chunk size (frames)')
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Loading model...")
    model = ComplexMTASSLightningStreaming.load_from_checkpoint(
        args.ckpt_path,
        model_class=Complex_MTASS_Streaming,
        loss_class=Complex_MTASS_model,
    )
    model.to(device)
    model.eval()
    model.freeze()
    print("Model loaded!")
    
    separator = RealTimeAudioSeparator(
        model,
        win_len=512,
        win_inc=256,
        fft_len=512,
        history_size=args.history_size,
        chunk_size=args.chunk_size,
        device=device
    )
    
    separator.process_file(args.input_wav, args.output_dir)


if __name__ == '__main__':
    main()
