#!/bin/bash
# Precision presets end to end on a backend with _bf16act kernels (CUDA):
#   1. bf16_mixed trains: exit 0, finite train loss that falls over 3 epochs;
#   2. a checkpoint saved under fp32 resumes under bf16_mixed, and one saved
#      under bf16_mixed resumes under fp32.
# Self-contained: generates a small corpus, a char tokenizer and the
# tokenized dataset under <tmp dir>.
#
# Usage: precision_e2e.sh <litnicelm binary> <backend plugin .so> <tmp dir>
set -u
BIN=$1
PLUGIN=$2
TMP=$3
rm -rf "$TMP"
mkdir -p "$TMP"
failures=0

pass() { echo "[PASS] $1"; }
fail() { echo "[FAIL] $1"; failures=$((failures + 1)); }

for i in $(seq 1 300); do
  echo "the quick brown fox $((i % 7)) jumps over the lazy dog $((i % 5))."
done > "$TMP/corpus.txt"

# config <name> <precision>: a 1-layer model; checkpoints and journal named
# after <name> so runs can share or keep them apart.
config() {
  cat > "$TMP/$1.yaml" <<EOF
conf:
  version: "1"
transformer_layers: 0
parameter_bytes: 0
optimizer_bytes: 0
arena_alignment: 64
max_steps: 1
precision: $2
backend:
  library: "$PLUGIN"
tokenizer:
  type: "only_seen_chars"
  target_vocab_size: 128
  training_corpus: "$TMP/corpus.txt"
  inter_file_boundary: ""
  artifacts_dir: "$TMP/tokenizer"
  bpe_vocab_file: ""
  bpe_merges_file: ""
  bpe_validation_num_threads: 1
  bpe_validation_sample_rate: 1
  run_validation: true
tokenization:
  input_corpus: "$TMP/corpus.txt"
  output_binary: "$TMP/corpus.bin"
  chunk_size_mb: 1
memory:
  alignment_bytes: 64
model:
  n_layers: 1
  n_heads: 2
  d_model: 32
  d_ff: 64
  max_seq_len: 16
model_algo:
  attention: "reference"
  ffn: "reference"
training:
  learning_rate: 0.01
  beta1: 0.9
  beta2: 0.999
  eps: 0.00000001
  weight_decay: 0.0
  incremental: false
  dry_run: false
  num_epochs_train: 3
  num_epochs_dry_run: 1
  save_interval_epochs: 1
  grad_clip: 1.0
  train_seq_len: 16
  window_stride: 16
  batch_size: 4
  target_loss: 0.0
  min_delta: 0.0
  patience_epochs: 0
  min_epochs: 0
  stop_on_nonfinite_loss: true
paths:
  model_file_latest: "$TMP/$3_latest.ckpt"
  model_file_best: "$TMP/$3_best.ckpt"
  journal_file: "$TMP/$3_journal.txt"
inference:
  prompt: "the "
  max_new: 16
  temp: 0.0
  top_k: 0
  top_p: 1.0
  seed: 12345
logging:
  show_bpe: false
  show_train: true
  show_inference: true
  report_every_n_steps: 1000
  epoch_report_every: 1
reporting:
  verbose_epoch_index: [0, 0]
  verbose_init: false
EOF
}

# Train loss of epoch <n> from a run's output, or empty.
epoch_loss() {
  grep -oE "train loss [0-9.eE+-]+ at epoch $2\b" "$1" | grep -oE '[0-9][0-9.eE+-]*' | head -1
}

config fp32_fresh fp32 fp32
config bf16_fresh bf16_mixed bf16
config bf16_from_fp32 bf16_mixed fp32
config fp32_from_bf16 fp32 bf16

if ! "$BIN" tokenizer_training --config "$TMP/fp32_fresh.yaml" > "$TMP/tok.out" 2>&1 ||
   ! "$BIN" encode --config "$TMP/fp32_fresh.yaml" > "$TMP/encode.out" 2>&1; then
  fail "tokenizer training and encoding (see $TMP/tok.out, $TMP/encode.out)"
  exit 1
fi

# 1. bf16_mixed trains.
"$BIN" train --config "$TMP/bf16_fresh.yaml" --incremental=false --epochs 3 \
  > "$TMP/bf16_fresh.out" 2>&1
rc=$?
first=$(epoch_loss "$TMP/bf16_fresh.out" 1)
last=$(epoch_loss "$TMP/bf16_fresh.out" 3)
if [ $rc -eq 0 ] && [ -n "$first" ] && [ -n "$last" ] &&
   awk -v a="$first" -v b="$last" 'BEGIN { exit !(b == b && b < a) }'; then
  pass "bf16_mixed trains: train loss $first (epoch 1) -> $last (epoch 3)"
else
  fail "bf16_mixed trains (rc=$rc, epoch 1 '$first', epoch 3 '$last'; see $TMP/bf16_fresh.out)"
fi

# 2. Cross-preset resume: 2 epochs under one preset, then epoch 3 under the
# other from the saved checkpoint.
resume() {  # <first config> <second config> <label>
  "$BIN" train --config "$TMP/$1.yaml" --incremental=false --epochs 2 \
    > "$TMP/$1.out" 2>&1
  local rc1=$?
  "$BIN" train --config "$TMP/$2.yaml" --incremental=true --epochs 3 \
    > "$TMP/$2.out" 2>&1
  local rc2=$?
  local loss
  loss=$(epoch_loss "$TMP/$2.out" 3)
  if [ $rc1 -eq 0 ] && [ $rc2 -eq 0 ] && grep -q "Resumed at" "$TMP/$2.out" &&
     [ -n "$loss" ]; then
    pass "$3: resumed and trained epoch 3 (train loss $loss)"
  else
    fail "$3 (rc $rc1/$rc2; see $TMP/$1.out, $TMP/$2.out)"
  fi
}
resume fp32_fresh bf16_from_fp32 "fp32 checkpoint resumes under bf16_mixed"
resume bf16_fresh fp32_from_bf16 "bf16_mixed checkpoint resumes under fp32"

if [ $failures -eq 0 ]; then
  echo "All precision end-to-end tests passed."
  exit 0
fi
echo "Precision end-to-end tests FAILED."
exit 1
