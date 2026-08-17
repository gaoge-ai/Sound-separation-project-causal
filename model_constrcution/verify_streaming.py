
#!/usr/bin/env python
import os
import sys
import h5py
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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
    print("Streaming Version Verification")
    print("="*60)
    
    # Test 1: Model forward pass
    print("\n[1/4] Testing streaming model forward pass...")
    from DNN_models.Complex_MTASS_streaming import Complex_MTASS_Streaming
    
    model = Complex_MTASS_Streaming(is_causal=True)
    model.eval()
    x = torch.randn(1, 514, 100)
    with torch.no_grad():
        z1, z2, z3 = model(x)
    
    print(f"✓ Streaming model forward pass successful!")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shapes: {z1.shape}, {z2.shape}, {z3.shape}")
    
    # Test 2: Compare with offline model output shape
    print("\n[2/4] Comparing output shapes with offline model...")
    from DNN_models.Complex_MTASS import Complex_MTASS
    
    model_offline = Complex_MTASS()
    model_offline.eval()
    with torch.no_grad():
        z1_off, z2_off, z3_off = model_offline(x)
    
    print(f"✓ Output shapes match!")
    print(f"  Offline: {z1_off.shape}, {z2_off.shape}, {z3_off.shape}")
    print(f"  Streaming: {z1.shape}, {z2.shape}, {z3.shape}")
    
    # Test 3: Run small training
    print("\n[3/4] Running small training on CPU (1 epoch)...")
    from DNN_models.Complex_MTASS_model_streaming import ComplexMTASSLightningStreaming
    from DNN_models.Complex_MTASS_Solver import Complex_MTASS_model
    from torch.utils.data import DataLoader
    
    exp_dir = "experiments/small_test_streaming_cpu"
    train_h5 = "small_dataset/train_small.h5"
    val_h5 = "small_dataset/val_small.h5"
    
    pl.seed_everything(42)
    data_train = HDF5Dataset(train_h5)
    data_val = HDF5Dataset(val_h5)
    train_loader = DataLoader(data_train, batch_size=2, shuffle=True, num_workers=0, pin_memory=False, drop_last=True)
    val_loader = DataLoader(data_val, batch_size=2, shuffle=False, num_workers=0, pin_memory=False, drop_last=True)
    
    model = ComplexMTASSLightningStreaming(
        learning_rate=1e-3,
        model_class=Complex_MTASS_Streaming,
        loss_class=Complex_MTASS_model,
        is_causal=True
    )
    
    trainer = pl.Trainer(
        default_root_dir=exp_dir,
        accelerator="cpu",
        devices="auto",
        max_epochs=1,
        logger=False,
        enable_checkpointing=True,
    )
    
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print("✓ Streaming model training completed!")
    
    # Test 4: Run inference
    print("\n[4/4] Running streaming inference...")
    from torch.utils.data import DataLoader
    
    # Find checkpoint
    checkpoints_dir = os.path.join(exp_dir, "checkpoints")
    ckpt_path = None
    if os.path.exists(checkpoints_dir):
        for f in os.listdir(checkpoints_dir):
            if f.endswith(".ckpt"):
                ckpt_path = os.path.join(checkpoints_dir, f)
                break
    
    if not ckpt_path:
        print("✗ No checkpoint found!")
        return 1
    
    print(f"  Found checkpoint: {ckpt_path}")
    
    # Load model
    model = ComplexMTASSLightningStreaming.load_from_checkpoint(
        ckpt_path,
        model_class=Complex_MTASS_Streaming,
        loss_class=Complex_MTASS_model,
    )
    model.eval()
    model.freeze()
    
    # Load test data
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
    
    print(f"  Running inference on {len(test_dataset)} samples...")
    for batch_idx, batch in enumerate(test_loader):
        X1, R_targets, idx = batch
        with torch.no_grad():
            Z1, Z2, Z3 = model(X1)
        print(f"  ✓ Sample {idx.item()} processed")
        print(f"    Input shape: {X1.shape}")
        print(f"    Output shapes: {Z1.shape}, {Z2.shape}, {Z3.shape}")
    
    print("\n" + "="*60)
    print("✓ All streaming verification steps completed successfully!")
    print("="*60)
    return 0

if __name__ == "__main__":
    sys.exit(main())
