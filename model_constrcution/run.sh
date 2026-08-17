#!/bin/bash

exp_dir=experiments/multi_head_separation_2-5mix_3class_overlap_nosing
resume_ckpt="${exp_dir}/checkpoints/last.ckpt"

python3 train.py $exp_dir \
  --resume_ckpt $resume_ckpt \
  --train_h5 "/work107/luoxiaoxue/workspace/Complex-MTASSNet/processed_data/3class_nosing/train_ready.h5" \
  --val_h5 "/work107/luoxiaoxue/workspace/Complex-MTASSNet/processed_data/3class_nosing/valid_ready.h5" \
  --use_cuda \
  --gpus 0 1 2 3 4 5 6 7

