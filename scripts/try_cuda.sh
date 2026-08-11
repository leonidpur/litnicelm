#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

missing=0
for cmd in cmake pkg-config; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Missing dependency: $cmd"
    missing=1
  fi
done

if ! command -v c++ >/dev/null 2>&1 && ! command -v g++ >/dev/null 2>&1; then
  echo "Missing dependency: C++ compiler (c++ or g++)"
  missing=1
fi

if ! pkg-config --exists sentencepiece 2>/dev/null; then
  echo "Missing dependency: sentencepiece pkg-config package"
  missing=1
fi

if ! command -v nvcc >/dev/null 2>&1; then
  echo "Missing dependency: nvcc"
  missing=1
fi

if command -v cmake >/dev/null 2>&1 &&
   ! cmake --version >/dev/null 2>&1; then
  echo "Dependency check failed: cmake exists but cannot run: $(command -v cmake)"
  echo "Fix the local CMake installation or put a working cmake earlier in PATH."
  missing=1
fi

if [[ "$missing" -ne 0 ]]; then
  cat <<'MSG'

On Ubuntu/Debian, install common dependencies with:
  sudo apt install cmake g++ pkg-config libsentencepiece-dev

Install the NVIDIA CUDA Toolkit separately for nvcc, CUDA runtime, and cuBLAS.
MSG
  exit 1
fi

mkdir -p data/local artifacts/local checkpoints/local journal/local conf/local build

corpus="data/local/abc.txt"
config="conf/local/try_cuda.yaml"

{
  for _ in $(seq 1 128); do
    printf 'a b c a b c a b c a b c\n'
  done
} > "$corpus"

cat > "$config" <<'YAML'
conf:
  version: "1"

transformer_layers: 0
parameter_bytes: 0
optimizer_bytes: 0
arena_alignment: 64
max_steps: 1

backend:
  library: "./build/liblitnice_backend_cublas_plugin.so"

tokenizer:
  type: "only_seen_chars"
  target_vocab_size: 128
  training_corpus: "data/local/abc.txt"
  inter_file_boundary: "\n<|book_boundary|>\n"
  artifacts_dir: "artifacts/local/abc_cuda"
  bpe_vocab_file: ""
  bpe_merges_file: ""
  bpe_validation_num_threads: 1
  bpe_validation_sample_rate: 50
  run_validation: true

tokenization:
  input_corpus: "data/local/abc.txt"
  output_binary: "artifacts/local/abc_cuda/tokenized_dataset.bin"
  chunk_size_mb: 16

memory:
  alignment_bytes: 64

model:
  n_layers: 2
  n_heads: 2
  d_model: 32
  d_ff: 64
  max_seq_len: 8

model_algo:
  attention: "fused_inplace"
  ffn: "inplace_fused_bias_relu"

training:
  learning_rate: 0.002
  beta1: 0.9
  beta2: 0.999
  eps: 0.00000001
  weight_decay: 0.01
  incremental: false
  dry_run: false
  num_epochs_train: 1
  num_epochs_dry_run: 1
  save_interval_epochs: 1
  grad_clip: 1.0
  train_seq_len: 4
  window_stride: 4
  batch_size: 2
  target_loss: -1.0
  min_delta: 0.0
  patience_epochs: 0
  min_epochs: 0
  stop_on_nonfinite_loss: true

paths:
  model_file_latest: "checkpoints/local/abc_cuda_latest.ckpt"
  model_file_best: "checkpoints/local/abc_cuda_best.ckpt"
  journal_file: "journal/local/abc_cuda_journal.txt"

inference:
  prompt: ""
  max_new: 32
  temp: 1.0
  top_k: 0
  top_p: 1.0
  seed: 12345

logging:
  show_bpe: true
  show_train: true
  show_inference: true
  report_every_n_steps: 1
  epoch_report_every: 1

reporting:
  verbose_epoch_index: [-1, 0]
  verbose_init: false
YAML

jobs="$(getconf _NPROCESSORS_ONLN 2>/dev/null || printf '2')"

echo "==> Building litnicelm with CUDA/cuBLAS plugin if available"
cmake -S . -B build
cmake --build build -j "$jobs"

if [[ ! -f build/liblitnice_backend_cublas_plugin.so ]]; then
  cat <<'MSG'
CUDA/cuBLAS plugin was not built:
  build/liblitnice_backend_cublas_plugin.so

Check that the CUDA Toolkit, nvcc, CUDA runtime, and cuBLAS are installed and
visible to CMake.
MSG
  exit 1
fi

echo "==> Training tokenizer"
./build/litnicelm tokenizer_training --config "$config"

echo "==> Tokenizing corpus"
./build/litnicelm encode --config "$config"

echo "==> Training one short CUDA/cuBLAS epoch"
./build/litnicelm train --config "$config" --incremental=false --epochs 1

echo "==> Running inference"
./build/litnicelm infer --config "$config" --prompt "a b c " --max_new 32

echo
echo "CUDA/cuBLAS try-run complete."
echo "Generated local files are under data/local, artifacts/local, checkpoints/local, journal/local, and conf/local."
