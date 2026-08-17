#!/bin/bash

for ckpt_path in experiments/small_test_streaming_cpu/checkpoints/epoch=*.ckpt; do

n_mix=2
output_dir="./test_results_chunk_streaming/${n_mix}mix"
test_h5="./small_dataset/test_small.h5"

echo "ckpt_path: ${ckpt_path}"
python3 test_chunk_streaming.py \
  --test_h5 $test_h5 \
  --ckpt_path $ckpt_path \
  --output_dir $output_dir \
  --num_sources $n_mix \
  --history_size 256 \
  --chunk_size 32 \

done

