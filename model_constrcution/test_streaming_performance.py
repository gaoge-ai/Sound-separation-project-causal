
#!/usr/bin/env python
import os
import sys
import time
import h5py
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from DNN_models.Complex_MTASS_streaming import Complex_MTASS_Streaming
from DNN_models.Complex_MTASS_model_streaming import ComplexMTASSLightningStreaming
from DNN_models.Complex_MTASS_Solver import Complex_MTASS_model


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


class RealTimeStreamingSeparator:
    def __init__(self, model, history_size=256, output_hop=16):
        self.model = model
        self.model.eval()
        self.history_size = history_size
        self.output_hop = output_hop
        self.input_buffer = None
        
    def reset(self):
        self.input_buffer = None
    
    def _init_buffer(self, first_frames):
        pad_size = self.history_size - first_frames.shape[-1]
        if pad_size > 0:
            padded = F.pad(first_frames, (pad_size, 0))
            self.input_buffer = padded
        else:
            self.input_buffer = first_frames[..., -self.history_size:]
    
    def process(self, new_frames):
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


def train_small_streaming_model(use_gpu=False, gpu_device=1):
    print("="*60)
    print("Step 1: Training small streaming model")
    print("="*60)
    
    exp_dir = "experiments/streaming_perf_test"
    train_h5 = "small_dataset/train_small.h5"
    val_h5 = "small_dataset/val_small.h5"
    
    # Check if we have an existing checkpoint
    checkpoints_dir = os.path.join(exp_dir, "checkpoints")
    existing_ckpt = None
    if os.path.exists(checkpoints_dir):
        for f in os.listdir(checkpoints_dir):
            if f.endswith(".ckpt"):
                existing_ckpt = os.path.join(checkpoints_dir, f)
                break
    
    if existing_ckpt:
        print(f"✓ Found existing checkpoint, skipping training: {existing_ckpt}")
        return existing_ckpt
    
    # Train new model
    pl.seed_everything(42)
    data_train = HDF5Dataset(train_h5)
    data_val = HDF5Dataset(val_h5)
    train_loader = DataLoader(data_train, batch_size=2, shuffle=True, 
                              num_workers=0, pin_memory=False, drop_last=True)
    val_loader = DataLoader(data_val, batch_size=2, shuffle=False,
                            num_workers=0, pin_memory=False, drop_last=True)
    
    model = ComplexMTASSLightningStreaming(
        learning_rate=1e-3,
        model_class=Complex_MTASS_Streaming,
        loss_class=Complex_MTASS_model,
        is_causal=True
    )
    
    # Check if CUDA_VISIBLE_DEVICES is set
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    
    if cuda_visible:
        # If CUDA_VISIBLE_DEVICES is set, just use 'auto' to let Lightning handle it
        devices_arg = "auto"
    else:
        # Otherwise use the specified gpu_device
        devices_arg = [gpu_device] if use_gpu else "auto"
    
    trainer = pl.Trainer(
        default_root_dir=exp_dir,
        accelerator="gpu" if use_gpu else "cpu",
        devices=devices_arg,
        max_epochs=2,
        logger=False,
        enable_checkpointing=True,
    )
    
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    ckpt_path = None
    if os.path.exists(checkpoints_dir):
        for f in os.listdir(checkpoints_dir):
            if f.endswith(".ckpt"):
                ckpt_path = os.path.join(checkpoints_dir, f)
                break
    
    print(f"✓ Training complete, checkpoint: {ckpt_path}")
    return ckpt_path


def test_streaming_performance(ckpt_path, use_gpu=False, gpu_device=1):
    print("\n" + "="*60)
    print("Step 2: Testing streaming performance")
    print("="*60)
    
    model = ComplexMTASSLightningStreaming.load_from_checkpoint(
        ckpt_path,
        model_class=Complex_MTASS_Streaming,
        loss_class=Complex_MTASS_model,
    )
    if use_gpu:
        # If CUDA_VISIBLE_DEVICES is set, just use cuda()
        import os
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            model = model.cuda()
        else:
            model = model.cuda(gpu_device)
    model.eval()
    model.freeze()
    
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
            return X1
    
    test_dataset = TestHDF5Dataset(test_h5)
    
    configs = [
        {"history_size": 256, "input_hop": 16, "name": "256/16"},
        {"history_size": 256, "input_hop": 32, "name": "256/32"},
        {"history_size": 512, "input_hop": 16, "name": "512/16"},
        {"history_size": 512, "input_hop": 32, "name": "512/32"},
    ]
    
    results = {}
    
    for config in configs:
        print(f"\nTesting config: {config['name']}")
        
        separator = RealTimeStreamingSeparator(
            model.model,
            history_size=config['history_size'],
            output_hop=config['input_hop']
        )
        
        first_packet_latencies = []
        process_times = []
        total_audio_duration = 0
        total_process_time = 0
        
        for sample_idx in range(len(test_dataset)):
            full_audio = test_dataset[sample_idx].unsqueeze(0)
            if use_gpu:
                import os
                if 'CUDA_VISIBLE_DEVICES' in os.environ:
                    full_audio = full_audio.cuda()
                else:
                    full_audio = full_audio.cuda(gpu_device)
            n_frames = full_audio.shape[-1]
            
            separator.reset()
            
            # 第一次输入 - 测量首包延迟
            first_input = full_audio[..., :config['input_hop']]
            start_time = time.time()
            _, _, _ = separator.process(first_input)
            first_latency = (time.time() - start_time) * 1000  # ms
            first_packet_latencies.append(first_latency)
            
            # 继续处理剩余部分 - 测量平均处理时间
            audio_idx = config['input_hop']
            while audio_idx < n_frames:
                end_idx = min(audio_idx + config['input_hop'], n_frames)
                new_frames = full_audio[..., audio_idx:end_idx]
                
                start_time = time.time()
                _, _, _ = separator.process(new_frames)
                process_time = time.time() - start_time
                
                process_times.append(process_time)
                total_process_time += process_time
                audio_duration = (end_idx - audio_idx) / 62.5  # seconds
                total_audio_duration += audio_duration
                
                audio_idx = end_idx
        
        # 计算指标
        avg_first_latency = sum(first_packet_latencies) / len(first_packet_latencies)
        avg_process_time = sum(process_times) / len(process_times) * 1000  # ms
        rtf = total_process_time / total_audio_duration if total_audio_duration > 0 else 0
        
        results[config['name']] = {
            "first_packet_latency_ms": avg_first_latency,
            "avg_process_time_ms": avg_process_time,
            "rtf": rtf,
            "total_audio_duration_s": total_audio_duration,
            "total_process_time_s": total_process_time,
        }
        
        print(f"  First packet latency: {avg_first_latency:.2f} ms")
        print(f"  Avg process time: {avg_process_time:.2f} ms")
        print(f"  RTF: {rtf:.4f}x")
    
    return results


def print_results_summary(results):
    print("\n" + "="*60)
    print("Performance Results Summary")
    print("="*60)
    
    print(f"\n{'Config':<15} {'First Latency(ms)':<20} {'Avg Process(ms)':<15} {'RTF':<10}")
    print("-"*60)
    
    for config_name, metrics in results.items():
        print(f"{config_name:<15} "
              f"{metrics['first_packet_latency_ms']:<20.2f} "
              f"{metrics['avg_process_time_ms']:<15.2f} "
              f"{metrics['rtf']:<10.4f}")
    
    print("\n" + "="*60)
    print("Key Metrics Explained:")
    print("="*60)
    print("- First packet latency: Time to get first output (ms)")
    print("- Avg process time: Average time per hop (ms)")
    print("- RTF (Real-Time Factor): Process time / Audio duration")
    print("  RTF < 1: Real-time capable")
    print("  RTF = 1: Just real-time")
    print("  RTF > 1: Cannot keep up with real-time")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU for testing")
    parser.add_argument("--gpu-device", type=int, default=1, help="GPU device index to use (default: 1)")
    args = parser.parse_args()
    
    print("="*60)
    print("Streaming Model Performance Test")
    print("="*60)
    print(f"Using GPU: {args.use_gpu}")
    if args.use_gpu:
        print(f"GPU device: {args.gpu_device}")
        import os
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    
    # Step 1: Train small model
    ckpt_path = train_small_streaming_model(use_gpu=args.use_gpu, gpu_device=args.gpu_device)
    
    # Step 2: Test performance
    results = test_streaming_performance(ckpt_path, use_gpu=args.use_gpu, gpu_device=args.gpu_device)
    
    # Step 3: Print summary
    print_results_summary(results)
    
    print("\n✓ All tests completed!")


if __name__ == "__main__":
    main()
