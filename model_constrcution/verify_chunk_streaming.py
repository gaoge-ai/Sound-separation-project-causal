
#!/usr/bin/env python
import os
import sys
import h5py
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class RealTimeStreamingSeparator:
    def __init__(self, model, history_size=256, chunk_size=32):
        self.model = model
        self.model.eval()
        self.history_size = history_size
        self.chunk_size = chunk_size
        self.input_buffer = None
        
    def reset(self):
        self.input_buffer = None
    
    def _init_buffer(self, first_frames):
        import torch.nn.functional as F
        pad_size = self.history_size - first_frames.shape[-1]
        if pad_size > 0:
            padded = F.pad(first_frames, (pad_size, 0))
            self.input_buffer = padded
        else:
            self.input_buffer = first_frames[..., -self.history_size:]
    
    def process_chunk(self, new_frames):
        if self.input_buffer is None:
            self._init_buffer(new_frames)
        else:
            self.input_buffer = torch.cat(
                [self.input_buffer, new_frames], dim=-1
            )[..., -self.history_size:]
        
        with torch.no_grad():
            z1, z2, z3 = self.model(self.input_buffer)
        
        out1 = z1[..., -new_frames.shape[-1]:]
        out2 = z2[..., -new_frames.shape[-1]:]
        out3 = z3[..., -new_frames.shape[-1]:]
        
        return out1, out2, out3
    
    def process_full(self, full_input):
        self.reset()
        total_frames = full_input.shape[-1]
        
        z1_full = torch.zeros_like(full_input)
        z2_full = torch.zeros_like(full_input)
        z3_full = torch.zeros_like(full_input)
        
        for i in range(0, total_frames, self.chunk_size):
            end_idx = min(i + self.chunk_size, total_frames)
            chunk_frames = full_input[..., i:end_idx]
            
            z1_chunk, z2_chunk, z3_chunk = self.process_chunk(chunk_frames)
            
            z1_full[..., i:end_idx] = z1_chunk
            z2_full[..., i:end_idx] = z2_chunk
            z3_full[..., i:end_idx] = z3_chunk
        
        return z1_full, z2_full, z3_full


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
        Y_targets = [torch.from_numpy(self.h5_file[f'Y{i}'][idx]).float() for i in range(1, 4)]
        R_targets = [torch.from_numpy(self.h5_file[f'R{i}'][idx]).float() for i in range(1, 4)]
        
        return (X1, *Y_targets, *R_targets)


def main():
    print("="*60)
    print("Chunk Streaming Inference Test")
    print("="*60)
    
    # Load model
    print("\n[1/5] Loading model from checkpoint...")
    from DNN_models.Complex_MTASS_model_streaming import ComplexMTASSLightningStreaming
    from DNN_models.Complex_MTASS_streaming import Complex_MTASS_Streaming
    from DNN_models.Complex_MTASS_Solver import Complex_MTASS_model
    
    exp_dir = "experiments/small_test_streaming_cpu"
    checkpoints_dir = os.path.join(exp_dir, "checkpoints")
    ckpt_path = None
    if os.path.exists(checkpoints_dir):
        for f in os.listdir(checkpoints_dir):
            if f.endswith(".ckpt"):
                ckpt_path = os.path.join(checkpoints_dir, f)
                break
    
    if not ckpt_path:
        print(" No checkpoint found!")
        return 1
    
    print(f"  Found checkpoint: {ckpt_path}")
    
    model = ComplexMTASSLightningStreaming.load_from_checkpoint(
        ckpt_path,
        model_class=Complex_MTASS_Streaming,
        loss_class=Complex_MTASS_model,
    )
    model.eval()
    model.freeze()
    print("  Model loaded successfully!")
    
    # Initialize streaming separator
    print("\n[2/5] Initializing streaming separator...")
    separator = RealTimeStreamingSeparator(model, history_size=256, chunk_size=32)
    print("  Streaming separator initialized!")
    print("    history_size=256, chunk_size=32")
    
    # Load test data
    print("\n[3/5] Loading test data...")
    test_h5 = "small_dataset/test_small.h5"
    
    class TestHDF5Dataset(Dataset):
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
    
    test_dataset = TestHDF5Dataset(test_h5)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    print(f"  Loaded {len(test_dataset)} test samples")
    
    # Test full offline inference for comparison
    print("\n[4/5] Running full offline inference for comparison...")
    for batch_idx, batch in enumerate(test_loader):
        X1, R_targets, idx = batch
        with torch.no_grad():
            Z1_full_off, Z2_full_off, Z3_full_off = model(X1)
        print(f"  Sample {idx.item()} - Offline output shapes: {Z1_full_off.shape}, {Z2_full_off.shape}, {Z3_full_off.shape}")
        if batch_idx >= 0:
            break
    
    # Test chunk streaming inference
    print("\n[5/5] Running chunk streaming inference...")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    for batch_idx, batch in enumerate(test_loader):
        X1, R_targets, idx = batch
        print(f"\n  Processing sample {idx.item()}...")
        print(f"    Input shape: {X1.shape}")
        
        Z1, Z2, Z3 = separator.process_full(X1)
        
        print(f"    Sample {idx.item()} processed successfully!")
        print(f"    Output shapes: {Z1.shape}, {Z2.shape}, {Z3.shape}")
        
        if batch_idx >= 1:
            break
    
    print("\n" + "="*60)
    print(" Chunk streaming inference test completed successfully!")
    print("="*60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
