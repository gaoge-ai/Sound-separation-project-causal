
#!/bin/bash

cd "$(dirname "$0")/model_constrcution"

echo "Running in directory: $(pwd)"
echo ""

python3 test_wav_streaming_sdr.py \
  --wav_dir ./test_2mix_wavs \
  --ckpt_path experiments/small_test_streaming_cpu/checkpoints/epoch=0-step=2.ckpt \
  --output_dir ./wav_streaming_results \
  --history_size 256 \
  --chunk_size 32 \
  --num_samples 5
