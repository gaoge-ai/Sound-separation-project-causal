
#!/bin/bash

cd "$(dirname "$0")"

echo "="*60
echo "WAV Chunk Streaming Separation with SDR Evaluation"
echo "="*60

CSV_PATH="../dataset/3class_data_nosing/metadata/test_2mix.csv"
WAV_DIR="./test_2mix_wavs"
CKPT_PATH="experiments/small_test_streaming_cpu/checkpoints/epoch=0-step=2.ckpt"
OUTPUT_DIR="./wav_streaming_results"
NUM_SAMPLES=5

echo ""
echo "[1/3] Generating test wav files from CSV..."
python3 generate_test_wavs.py \
  --csv_path "$CSV_PATH" \
  --output_dir "$WAV_DIR" \
  --num_samples $NUM_SAMPLES

echo ""
echo "[2/3] Running chunk streaming separation and calculating SDR..."
python3 test_wav_streaming_sdr.py \
  --wav_dir "$WAV_DIR" \
  --ckpt_path "$CKPT_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --history_size 256 \
  --chunk_size 32 \
  --num_samples $NUM_SAMPLES

echo ""
echo "[3/3] Done!"
echo ""
echo "Output files:"
echo "  - Test wavs: $WAV_DIR"
echo "  - Separation results: $OUTPUT_DIR"
echo ""
echo "="*60
