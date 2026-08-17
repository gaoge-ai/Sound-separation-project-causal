#!/bin/bash

for ckpt_path in experiments/multi_head_separation_2-5mix_3class_overlap_nosing/checkpoints/epoch=0036*.ckpt; do

n_mix=2
output_dir="/work107/luoxiaoxue/workspace/Complex-MTASSNet/test_results/3class_nosing/${n_mix}mix"
test_h5="/work107/luoxiaoxue/workspace/Complex-MTASSNet/processed_data/3class_nosing/test_${n_mix}mix_ready.h5"

echo "ckpt_path: ${ckpt_path}"
python3 test.py \
  --test_h5 $test_h5 \
  --ckpt_path $ckpt_path \
  --output_dir $output_dir \
  --num_sources $n_mix \
  --use_cuda \

done

