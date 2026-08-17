
#!/usr/bin/env python3
import os
import sys

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_constrcution'))

print(f"Running in: {os.getcwd()}")
print()

import subprocess

cmd = [
    sys.executable, 'test_wav_streaming_sdr.py',
    '--wav_dir', './test_2mix_wavs',
    '--ckpt_path', 'experiments/small_test_streaming_cpu/checkpoints/epoch=0-step=2.ckpt',
    '--output_dir', './wav_streaming_results',
    '--history_size', '256',
    '--chunk_size', '32',
    '--num_samples', '5'
]

print("Running command:")
print(' '.join(cmd))
print()

result = subprocess.run(cmd, capture_output=False, text=True)
