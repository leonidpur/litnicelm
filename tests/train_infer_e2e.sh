#!/bin/bash
# Training -> inference end to end on CUDA, for every precision preset and
# position encoding:
#
#   {fp32, bf16_mixed} x {rope, learned} x {fresh, resume}
#
# Each training run: exit 0; train and val loss finite every epoch; train
# loss falls from epoch 1 to 3; final train and val loss < 0.5 * ln(V).
# Resume runs (2 epochs, then --incremental to 3) must resume at exactly the
# step where the first run stopped, end at the fresh run's final step, and
# match the fresh run's epoch-3 train loss within TOL_RESUME. The LR is
# constant after a warmup that spans the resume point, so epoch 3 depends on
# the restored step, LR schedule, optimizer state and data position. (Under
# cosine a 2-epoch invocation plans a shorter schedule than a 3-epoch one, so
# the two would not match by design.)
#
# Each training run is followed by greedy inference (temp 0): exit 0;
# exactly max_new generated characters (char tokenizer); only corpus
# characters; the same command twice gives identical text; at least one
# corpus word. Across presets: an fp32 checkpoint resumes and infers under
# bf16_mixed, and a bf16_mixed checkpoint under fp32. The fp32 fresh runs are
# repeated on the CPU backend; final train loss within TOL_CPU of CUDA.
#
# Tolerances, from 10 runs of the whole test (MEASURE=1 prints every compared
# quantity; 2026-10-22, RTX 4070 Laptop). CUDA atomics make runs
# non-bit-exact: the fresh epoch-3 train loss alone varies by up to 0.016
# between runs.
#   TOL_RESUME 0.05   observed |resume - fresh| <= 0.0168 (all cases).
#                     Fault check: zeroing the AdamW moments on resume gives
#                     0.085..0.151, caught. Not restoring the step gives only
#                     0.016..0.031, inside the noise: the exact step checks
#                     catch that one.
#   TOL_CPU    0.035  observed |CPU - CUDA| fp32 final train loss <= 0.0113.
#   Loss limit 0.5 ln V: observed final train and val loss <= 0.121 of it.
#
# Self-contained: generates a small corpus, a char tokenizer and the
# tokenized dataset under <tmp dir>, which keeps every run's output (the
# next run clears it). Exits 77 (ctest: skipped) without a CUDA plugin.
#
# Usage: train_infer_e2e.sh <litnicelm> <CUDA plugin .so or ""> <CPU plugin .so>
#                           <tmp dir>
set -u
BIN=$1
CUDA=$2
CPU=$3
TMP=$4
TOL_RESUME=${TOL_RESUME:-0.05}
TOL_CPU=${TOL_CPU:-0.035}
MAX_NEW=40
PROMPT="the "

if [ -z "$CUDA" ] || [ ! -f "$CUDA" ]; then
  echo "SKIPPED: no CUDA plugin built (bf16_mixed needs the _bf16act kernels)"
  exit 77
fi
rm -rf "$TMP"
mkdir -p "$TMP"
failures=0
started=$(date +%s)

pass() { echo "[PASS] $1"; }
fail() {
  echo "[FAIL] $1"
  shift
  for f in "$@"; do echo "       see $f"; done
  failures=$((failures + 1))
}
measure() { [ -n "${MEASURE:-}" ] && echo "MEASURE $*"; }
# abs(a - b) <= tol
within() { awk -v a="$1" -v b="$2" -v t="$3" 'BEGIN { d = a - b; if (d < 0) d = -d; exit !(d <= t) }'; }
absdiff() { awk -v a="$1" -v b="$2" 'BEGIN { d = a - b; if (d < 0) d = -d; printf "%.6f", d }'; }

words="the quick brown fox jumps over lazy dog cat sleeps under warm sun"
for i in $(seq 1 400); do
  set -- $words
  shift $((i % 5))
  echo "the $1 $2 $3 $4 near the old mill $((i % 7))."
done > "$TMP/corpus.txt"
# Corpus alphabet and words, for the inference checks.
alphabet=$(fold -w1 "$TMP/corpus.txt" | sort -u | tr -d '\n')
tr -cs 'a-z' '\n' < "$TMP/corpus.txt" | awk 'length >= 3' | sort -u > "$TMP/words.txt"

# config <name> <precision> <position encoding> <checkpoint prefix> <plugin>
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
  library: "$5"
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
  position_encoding: "$3"
training:
  learning_rate: 0.01
  lr_schedule: "constant"
  warmup_steps: $WARMUP
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
  validation_fraction: 0.1
  validation_chunk_tokens: 256
  validation_every_epochs: 1
paths:
  model_file_latest: "$TMP/$4_latest.ckpt"
  model_file_best: "$TMP/$4_best.ckpt"
  journal_file: "$TMP/$4_journal.txt"
inference:
  prompt: "$PROMPT"
  max_new: $MAX_NEW
  temp: 0.0
  top_k: 0
  top_p: 1.0
  seed: 12345
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
}

# epoch_field <output> <epoch> <train|val>: that epoch's loss, or empty.
epoch_field() {
  grep -E "Validation epoch $2:" "$1" | head -1 |
    sed -nE "s/.* $3 ([^ ,|]+).*/\1/p"
}
# final_step <output>: the step of the run's last saved checkpoint.
final_step() {
  sed -nE 's/.*Saved latest: .*\(epoch [0-9]+, step ([0-9]+),.*/\1/p' "$1" | tail -1
}
# check_steps <label> <first output> <resumed output>: the resumed run starts
# where the first stopped (2 epochs) and ends at 3 epochs, exactly.
check_steps() {
  local want_resume=$((2 * STEPS_PER_EPOCH)) want_end=$((3 * STEPS_PER_EPOCH))
  local stopped resumed ended
  stopped=$(final_step "$2")
  resumed=$(sed -nE 's/.*Resumed at Step \[([0-9]+)\/.*/\1/p' "$3" | head -1)
  ended=$(final_step "$3")
  if [ "$stopped" = "$want_resume" ] && [ "$resumed" = "$want_resume" ] &&
     [ "$ended" = "$want_end" ]; then
    pass "$1: stopped at step $stopped, resumed at $resumed, ended at $ended"
    return 0
  fi
  fail "$1: steps stopped $stopped, resumed '$resumed', ended $ended (want $want_resume, $want_resume, $want_end)" "$2" "$3"
  return 1
}
is_finite() { [[ "$1" =~ ^[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?$ ]]; }

# check_training <label> <ln V limit> <epochs> <output for epoch 1> ...
# Epoch k's loss is read from the k-th output file given.
check_training() {
  local label=$1 limit=$2 epochs=$3
  shift 3
  local outs=("$@") ok=1 e train val first last last_val
  for ((e = 1; e <= epochs; e++)); do
    train=$(epoch_field "${outs[$((e - 1))]}" "$e" train)
    val=$(epoch_field "${outs[$((e - 1))]}" "$e" val)
    if ! is_finite "$train" || ! is_finite "$val"; then
      fail "$label: epoch $e loss not finite (train '$train', val '$val')" "${outs[$((e - 1))]}"
      return 1
    fi
    [ "$e" -eq 1 ] && first=$train
    last=$train
    last_val=$val
  done
  if ! awk -v a="$first" -v b="$last" 'BEGIN { exit !(b < a) }'; then
    fail "$label: train loss did not fall ($first -> $last)" "${outs[@]}"
    ok=0
  fi
  if ! awk -v t="$last" -v v="$last_val" -v l="$limit" 'BEGIN { exit !(t < l && v < l) }'; then
    fail "$label: final loss not below 0.5 ln V = $limit (train $last, val $last_val)" "${outs[@]}"
    ok=0
  fi
  measure "final_train_over_limit ${label// /_} $(awk -v t="$last" -v l="$limit" 'BEGIN { printf "%.4f", t / l }')"
  [ $ok -eq 1 ] && pass "$label: trains, loss $first -> $last (val $last_val, limit $limit)"
  return $((1 - ok))
}

# generated_text <infer output>: the text after the end line, prompt removed.
generated_text() {
  sed -n '/^\[Inference\]\[End\]/,$p' "$1" | tail -n +2 | sed "1s/^$PROMPT//"
}

# check_inference <label> <config>
check_inference() {
  local label=$1 cfg=$2 a="$TMP/infer_$1_a.out" b="$TMP/infer_$1_b.out" text n
  "$BIN" infer --config "$cfg" > "$a" 2>&1
  local rc1=$?
  "$BIN" infer --config "$cfg" > "$b" 2>&1
  local rc2=$?
  if [ $rc1 -ne 0 ] || [ $rc2 -ne 0 ]; then
    fail "$label inference: exit $rc1/$rc2" "$a" "$b"
    return
  fi
  text=$(generated_text "$a"; echo x)
  text=${text%x}
  text=${text%$'\n'}
  n=${#text}
  if ! sed -n '/^\[Inference\]\[End\]/,$p' "$a" | tail -n +2 | head -1 | grep -q "^$PROMPT"; then
    fail "$label inference: output does not start with the prompt" "$a"
  elif [ "$n" -ne "$MAX_NEW" ]; then
    fail "$label inference: $n generated characters, expected $MAX_NEW" "$a"
  elif printf '%s' "$text" | tr -d "$alphabet"$'\n' | grep -q .; then
    fail "$label inference: characters outside the corpus alphabet" "$a"
  elif [ "$(generated_text "$a")" != "$(generated_text "$b")" ]; then
    fail "$label inference: two identical runs differ" "$a" "$b"
  elif ! printf '%s\n' "$text" | tr -cs 'a-z' '\n' | grep -qxF -f "$TMP/words.txt"; then
    fail "$label inference: no corpus word in the generated text" "$a"
  else
    pass "$label inference: \"$(printf '%s' "$text" | tr '\n' '|')\""
  fi
}

# Tokenizer and dataset; the vocabulary size V sets the loss limit.
WARMUP=0
config base fp32 rope base "$CUDA"
if ! "$BIN" tokenizer_training --config "$TMP/base.yaml" > "$TMP/tok.out" 2>&1 ||
   ! "$BIN" encode --config "$TMP/base.yaml" > "$TMP/encode.out" 2>&1; then
  fail "tokenizer training and encoding" "$TMP/tok.out" "$TMP/encode.out"
  exit 1
fi
"$BIN" dry_run --config "$TMP/base.yaml" > "$TMP/dry.out" 2>&1
V=$(sed -nE 's/.*model.vocab_size set to ([0-9]+).*/\1/p' "$TMP/dry.out" | head -1)
STEPS=$(sed -nE 's/.*Step \[[0-9]+\/([0-9]+)\].*/\1/p' "$TMP/dry.out" | head -1)
if [ -z "$V" ]; then
  fail "vocabulary size not found" "$TMP/dry.out"
  exit 1
fi
LIMIT=$(awk -v v="$V" 'BEGIN { printf "%.4f", 0.5 * log(v) }')
echo "corpus: V=$V, 0.5 ln V = $LIMIT"

# Warmup to 2.5 epochs: the resume point (end of epoch 2) is mid-warmup.
STEPS_PER_EPOCH=$(sed -nE 's/.*batches_per_epoch[=: ]+([0-9]+).*/\1/p' "$TMP/dry.out" | head -1)
if [ -z "$STEPS_PER_EPOCH" ]; then
  "$BIN" train --config "$TMP/base.yaml" --incremental=false --epochs 1 > "$TMP/probe.out" 2>&1
  STEPS_PER_EPOCH=$(sed -nE 's/.*\(epoch 1, step ([0-9]+),.*/\1/p' "$TMP/probe.out" | head -1)
fi
WARMUP=$((STEPS_PER_EPOCH * 5 / 2))
echo "steps per epoch: $STEPS_PER_EPOCH, warmup: $WARMUP steps"

declare -A fresh_loss
for prec in fp32 bf16_mixed; do
  for pos in rope learned; do
    case_=${prec}_${pos}
    # Fresh: 3 epochs.
    config "${case_}_fresh" "$prec" "$pos" "${case_}_fresh" "$CUDA"
    out="$TMP/${case_}_fresh.out"
    "$BIN" train --config "$TMP/${case_}_fresh.yaml" --incremental=false --epochs 3 > "$out" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
      fail "$case_ fresh: exit $rc" "$out"
    elif check_training "$case_ fresh" "$LIMIT" 3 "$out" "$out" "$out"; then
      fresh_loss[$case_]=$(epoch_field "$out" 3 train)
      check_inference "${case_}_fresh" "$TMP/${case_}_fresh.yaml"
    fi
    # Resume: 2 epochs, then to 3 from the checkpoint.
    config "${case_}_resume" "$prec" "$pos" "${case_}_resume" "$CUDA"
    out1="$TMP/${case_}_resume_1.out"
    out2="$TMP/${case_}_resume_2.out"
    "$BIN" train --config "$TMP/${case_}_resume.yaml" --incremental=false --epochs 2 > "$out1" 2>&1
    rc1=$?
    "$BIN" train --config "$TMP/${case_}_resume.yaml" --incremental=true --epochs 3 > "$out2" 2>&1
    rc2=$?
    if [ $rc1 -ne 0 ] || [ $rc2 -ne 0 ]; then
      fail "$case_ resume: exit $rc1/$rc2" "$out1" "$out2"
    elif ! check_steps "$case_ resume" "$out1" "$out2"; then
      :
    elif check_training "$case_ resume" "$LIMIT" 3 "$out1" "$out1" "$out2"; then
      resumed=$(epoch_field "$out2" 3 train)
      if [ -n "${fresh_loss[$case_]:-}" ]; then
        d=$(absdiff "$resumed" "${fresh_loss[$case_]}")
        measure "resume_delta $case_ $d"
        if within "$resumed" "${fresh_loss[$case_]}" "$TOL_RESUME"; then
          pass "$case_ resume: epoch-3 train loss $resumed matches fresh ${fresh_loss[$case_]} (|d| $d <= $TOL_RESUME)"
        else
          fail "$case_ resume: epoch-3 train loss $resumed vs fresh ${fresh_loss[$case_]} (|d| $d > $TOL_RESUME)" "$out2" "$TMP/${case_}_fresh.out"
        fi
      fi
      check_inference "${case_}_resume" "$TMP/${case_}_resume.yaml"
    fi
  done
done

# Across presets: resume and infer from the other preset's checkpoint.
cross() {  # <from preset> <to preset>
  local from=$1 to=$2 name="${1}_to_${2}"
  config "${name}_a" "$from" rope "$name" "$CUDA"
  config "${name}_b" "$to" rope "$name" "$CUDA"
  "$BIN" train --config "$TMP/${name}_a.yaml" --incremental=false --epochs 2 > "$TMP/${name}_1.out" 2>&1
  local rc1=$?
  "$BIN" train --config "$TMP/${name}_b.yaml" --incremental=true --epochs 3 > "$TMP/${name}_2.out" 2>&1
  local rc2=$?
  if [ $rc1 -ne 0 ] || [ $rc2 -ne 0 ]; then
    fail "$from checkpoint resumes under $to (exit $rc1/$rc2)" "$TMP/${name}_1.out" "$TMP/${name}_2.out"
  elif check_steps "$from checkpoint resumed under $to" "$TMP/${name}_1.out" "$TMP/${name}_2.out"; then
    check_training "$from checkpoint resumed under $to" "$LIMIT" 3 \
      "$TMP/${name}_1.out" "$TMP/${name}_1.out" "$TMP/${name}_2.out"
  fi
  config "${name}_infer" "$to" rope "${from}_rope_fresh" "$CUDA"
  check_inference "${from}_checkpoint_under_$to" "$TMP/${name}_infer.yaml"
}
cross fp32 bf16_mixed
cross bf16_mixed fp32

# CPU backend: fp32 fresh runs agree with CUDA.
if [ -n "$CPU" ] && [ -f "$CPU" ]; then
  for pos in rope learned; do
    config "cpu_fp32_${pos}" fp32 "$pos" "cpu_fp32_${pos}" "$CPU"
    out="$TMP/cpu_fp32_${pos}.out"
    "$BIN" train --config "$TMP/cpu_fp32_${pos}.yaml" --incremental=false --epochs 3 > "$out" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
      fail "CPU fp32 $pos: exit $rc" "$out"
    elif check_training "CPU fp32 $pos" "$LIMIT" 3 "$out" "$out" "$out"; then
      cpu=$(epoch_field "$out" 3 train)
      cuda=${fresh_loss[fp32_$pos]:-}
      if [ -n "$cuda" ]; then
        d=$(absdiff "$cpu" "$cuda")
        measure "cpu_delta $pos $d"
        if within "$cpu" "$cuda" "$TOL_CPU"; then
          pass "CPU fp32 $pos: final train loss $cpu matches CUDA $cuda (|d| $d <= $TOL_CPU)"
        else
          fail "CPU fp32 $pos: final train loss $cpu vs CUDA $cuda (|d| $d > $TOL_CPU)" "$out" "$TMP/fp32_${pos}_fresh.out"
        fi
      fi
    fi
  done
else
  echo "[SKIP] CPU comparison: no CPU plugin given"
fi

echo "elapsed: $(($(date +%s) - started)) s"
if [ $failures -eq 0 ]; then
  echo "All training -> inference end-to-end tests passed."
  exit 0
fi
echo "Training -> inference end-to-end tests FAILED ($failures); outputs kept in $TMP"
exit 1
