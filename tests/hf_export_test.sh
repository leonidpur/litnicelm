#!/bin/bash
# Export end to end on the CPU backend: trains a tiny model for one epoch
# with learned and with rope positions, runs `litnicelm export` on each, and
# checks the export directory against the checkpoint with
# litnice_hf_export_test. Self-contained (generated corpus, char tokenizer).
#
# Usage: hf_export_test.sh <litnicelm> <litnice_hf_export_test> <CPU plugin .so>
#                          <tmp dir>
set -u
BIN=$1
CHECK=$2
CPU=$3
TMP=$4
rm -rf "$TMP"
mkdir -p "$TMP"
failures=0

for i in $(seq 1 60); do
  echo "the quick brown fox $((i % 7)) jumps over the lazy dog $((i % 5))."
done > "$TMP/corpus.txt"

for pos in learned rope; do
  cat > "$TMP/$pos.yaml" <<EOF
conf:
  version: "1"
transformer_layers: 0
parameter_bytes: 0
optimizer_bytes: 0
arena_alignment: 64
max_steps: 1
backend:
  library: "$CPU"
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
  n_layers: 2
  n_heads: 2
  d_model: 16
  d_ff: 32
  max_seq_len: 16
model_algo:
  attention: "reference"
  ffn: "reference"
  position_encoding: "$pos"
training:
  learning_rate: 0.01
  beta1: 0.9
  beta2: 0.999
  eps: 0.00000001
  weight_decay: 0.0
  incremental: false
  dry_run: false
  num_epochs_train: 1
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
  model_file_latest: "$TMP/${pos}_latest.ckpt"
  model_file_best: "$TMP/${pos}_best.ckpt"
  journal_file: "$TMP/${pos}_journal.txt"
inference:
  prompt: "the "
  max_new: 8
  temp: 0.0
  top_k: 0
  top_p: 1.0
  seed: 1
logging:
  show_bpe: false
  show_train: true
  show_inference: true
  report_every_n_steps: 100000
  epoch_report_every: 1
reporting:
  verbose_epoch_index: [0, 0]
  verbose_init: false
EOF
  if [ "$pos" = learned ]; then
    if ! "$BIN" tokenizer_training --config "$TMP/$pos.yaml" > "$TMP/tok.out" 2>&1 ||
       ! "$BIN" encode --config "$TMP/$pos.yaml" > "$TMP/encode.out" 2>&1; then
      echo "[FAIL] tokenizer training and encoding (see $TMP/tok.out, $TMP/encode.out)"
      exit 1
    fi
  fi
  if ! "$BIN" train --config "$TMP/$pos.yaml" > "$TMP/${pos}_train.out" 2>&1; then
    echo "[FAIL] $pos: training (see $TMP/${pos}_train.out)"
    failures=$((failures + 1))
    continue
  fi
  if ! "$BIN" export --config "$TMP/$pos.yaml" --out "$TMP/export_$pos" > "$TMP/${pos}_export.out" 2>&1; then
    echo "[FAIL] $pos: export (see $TMP/${pos}_export.out)"
    failures=$((failures + 1))
    continue
  fi
  echo "== $pos positions"
  "$CHECK" "$TMP/$pos.yaml" "$TMP/export_$pos" 2>&1 | grep -E '^\[|tests'
  [ "${PIPESTATUS[0]}" -eq 0 ] || failures=$((failures + 1))
done

if [ $failures -eq 0 ]; then
  echo "All export end-to-end tests passed."
  exit 0
fi
echo "Export end-to-end tests FAILED ($failures); outputs kept in $TMP"
exit 1
