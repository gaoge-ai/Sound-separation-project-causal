
#!/bin/bash

cd "$(dirname "$0")"

echo "="*60
echo "Audio Streaming Separation Pipeline"
echo "="*60

echo ""
echo "[1/3] Extracting test wav file from h5 dataset..."
python3 extract_test_wav.py

echo ""
echo "[2/3] Running audio streaming separation..."
python3 test_audio_streaming.py \
  --input_wav ./test_wav/mixture.wav \
  --ckpt_path experiments/small_test_streaming_cpu/checkpoints/epoch=0-step=2.ckpt \
  --output_dir ./audio_streaming_results \
  --history_size 256 \
  --chunk_size 32

echo ""
echo "[3/3] Done!"
echo ""
echo "Output files:"
echo "  - audio_streaming_results/mixture.wav (input)"
echo "  - audio_streaming_results/speech.wav (separated)"
echo "  - audio_streaming_results/music.wav (separated)"
echo "  - audio_streaming_results/others.wav (separated)"
echo ""
echo "="*60
