#!/bin/bash
# Training -> inference end to end on CUDA, for every precision preset and
# position encoding:
#
#   {fp32, bf16_mixed} x {rope, learned} x {fresh, resume}
#
# The corpus is one fixed text repeated, in which every 8-character window
# is unique (checked here), so each next character is determined by the 8
# before it. The models memorize it, and greedy inference from a prompt cut
# from the text must reproduce the text's continuation exactly.
#
# Each training run: exit 0; train and val loss finite every epoch; train
# loss falls from the first to the last epoch; final train and val loss
# < 0.5 * ln(V). (The loss does not reach 0: each 16-token training window
# starts with positions whose context is shorter than 8 characters, where
# the next character is ambiguous.) The LR follows a cosine schedule over
# EPOCHS; a decaying LR is what makes the memorization reliable. Resume runs
# are the same plan interrupted (--stop_after_epoch EPOCHS-1) and continued
# with --incremental: they must resume at exactly the step where the first
# leg stopped, end at the fresh run's final step (step counter and LR
# schedule position), and match the fresh run's final train loss within
# TOL_RESUME (weights, optimizer state, data position).
#
# Each training run is followed by greedy inference: exit 0; exactly MAX_NEW
# generated characters; the same command twice gives identical text; the
# text equals the corpus continuation of the prompt, character for
# character; the smallest top-1 minus top-2 probability over the generated
# characters (inference.report_margin) is at least MIN_MARGIN. Across
# presets: an fp32 checkpoint resumes and infers under bf16_mixed, and a
# bf16_mixed checkpoint under fp32. The fp32 fresh runs are repeated on the
# CPU backend (in the background, in parallel with the CUDA runs): same
# inference checks, final train loss within TOL_CPU of CUDA. Negative
# control: an untrained checkpoint (learning rate 1e-9) must miss the
# expected text in at least NEG_MIN_WRONG of MAX_NEW characters.
#
# Tolerances, from 10 runs of the whole test (MEASURE=1 prints every compared
# quantity; 2026-10-22, RTX 4070 Laptop; CUDA atomics make runs
# non-bit-exact):
#   MIN_MARGIN    0.5    observed min top-1 margin 0.871 over all 160
#                        inference checks; the untrained control misses 38/40.
#   TOL_RESUME    0.008  observed |resume - fresh| <= 0.0042. Fault check:
#                        zeroing the AdamW moments on resume gives
#                        0.0149..0.0201 (inference stays exact, so only this
#                        check sees it); not restoring the step fails the
#                        exact step checks.
#   TOL_CPU       0.006  observed |CPU - CUDA| fp32 final loss <= 0.0017.
#   0.5 ln V      observed final train/val loss <= 0.123 of it.
#
# Self-contained: generates the corpus, a char tokenizer and the tokenized
# dataset under <tmp dir>, which keeps every run's output (the next run
# clears it). Exits 77 (ctest: skipped) without a CUDA plugin.
#
# Usage: train_infer_e2e.sh <litnicelm> <CUDA plugin .so or ""> <CPU plugin .so>
#                           <tmp dir>
set -u
BIN=$1
CUDA=$2
CPU=$3
TMP=$4
TOL_RESUME=${TOL_RESUME:-0.008}
TOL_CPU=${TOL_CPU:-0.006}
MIN_MARGIN=${MIN_MARGIN:-0.5}
NEG_MIN_WRONG=10
EPOCHS=10
MAX_NEW=40
WINDOW=8

TEXT="a quick brown fox jumps over seven lazy dogs while twelve wizards box \
nimbly near the old mill. pale moonlight drifts across frozen valleys and \
quiet harbors wake slowly beneath grey skies. hungry gulls circle fishing \
boats returning at dawn with silver catch. "
PROMPT_START="quiet harbor"

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

# The text read cyclically (its copies are joined end to start).
CYCLIC="$TEXT$TEXT"
PROMPT=$PROMPT_START
prompt_at=${TEXT%%"$PROMPT"*}
prompt_at=${#prompt_at}
EXPECTED=${CYCLIC:$((prompt_at + ${#PROMPT})):$MAX_NEW}
duplicate=$(awk -v t="$TEXT" -v w="$WINDOW" 'BEGIN {
  d = t substr(t, 1, w - 1)
  for (i = 1; i <= length(t); ++i) {
    g = substr(d, i, w)
    if (g in seen) { printf "\"%s\" at %d and %d", g, seen[g] - 1, i - 1; exit }
    seen[g] = i
  }
}')
if [ -n "$duplicate" ]; then
  fail "the text has a repeated $WINDOW-character window: $duplicate"
  exit 1
fi
pass "every $WINDOW-character window of the ${#TEXT}-character text is unique (cyclically)"
echo "prompt \"$PROMPT\" -> expected \"$EXPECTED\""
for i in $(seq 1 64); do printf '%s' "$TEXT"; done > "$TMP/corpus.txt"

# config <name> <precision> <position encoding> <checkpoint prefix> <plugin>
#        [learning rate]
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
  learning_rate: ${6:-0.005}
  lr_schedule: "cosine"
  min_lr_ratio: 0.02
  warmup_steps: $WARMUP
  beta1: 0.9
  beta2: 0.999
  eps: 0.00000001
  weight_decay: 0.0
  incremental: false
  dry_run: false
  num_epochs_train: $EPOCHS
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
  report_margin: true
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
is_finite() { [[ "$1" =~ ^[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?$ ]]; }

# check_training <label> <output of epochs 1..split> <split> <output of the rest>
check_training() {
  local label=$1 first_out=$2 split=$3 rest_out=$4 ok=1 e out train val first last last_val
  for ((e = 1; e <= EPOCHS; e++)); do
    out=$first_out
    [ "$e" -gt "$split" ] && out=$rest_out
    train=$(epoch_field "$out" "$e" train)
    val=$(epoch_field "$out" "$e" val)
    if ! is_finite "$train" || ! is_finite "$val"; then
      fail "$label: epoch $e loss not finite (train '$train', val '$val')" "$out"
      return 1
    fi
    [ "$e" -eq 1 ] && first=$train
    last=$train
    last_val=$val
  done
  if ! awk -v a="$first" -v b="$last" 'BEGIN { exit !(b < a) }'; then
    fail "$label: train loss did not fall ($first -> $last)" "$first_out" "$rest_out"
    ok=0
  fi
  if ! awk -v t="$last" -v v="$last_val" -v l="$LIMIT" 'BEGIN { exit !(t < l && v < l) }'; then
    fail "$label: final loss not below 0.5 ln V = $LIMIT (train $last, val $last_val)" "$first_out" "$rest_out"
    ok=0
  fi
  measure "final_loss_over_limit ${label// /_} $(awk -v t="$last" -v v="$last_val" -v l="$LIMIT" 'BEGIN { printf "%.4f", (t > v ? t : v) / l }')"
  [ $ok -eq 1 ] && pass "$label: trains, loss $first -> $last (val $last_val, limit $LIMIT)"
  return $((1 - ok))
}

# check_steps <label> <first output> <resumed output>: the resumed run starts
# where the first stopped and ends where the fresh run ends, exactly.
check_steps() {
  local want_resume=$(((EPOCHS - 1) * STEPS_PER_EPOCH)) want_end=$((EPOCHS * STEPS_PER_EPOCH))
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

# generated_text <infer output>: the text after the end line, prompt removed.
generated_text() {
  local full
  full=$(sed -n '/^\[Inference\]\[End\]/,$p' "$1" | tail -n +2; echo x)
  full=${full%x}
  full=${full%$'\n'}
  printf '%s' "${full#"$PROMPT"}"
}
# wrong_chars <a> <b>: positions where two equal-length strings differ.
wrong_chars() {
  local i n=0
  for ((i = 0; i < ${#1}; ++i)); do
    [ "${1:i:1}" != "${2:i:1}" ] && n=$((n + 1))
  done
  echo $n
}

# infer_twice <label> <config>: runs inference twice into
# infer_<label>_{a,b}.out; returns 1 (after reporting) if either fails.
infer_twice() {
  local a="$TMP/infer_$1_a.out" b="$TMP/infer_$1_b.out"
  "$BIN" infer --config "$2" > "$a" 2>&1
  local rc1=$?
  "$BIN" infer --config "$2" > "$b" 2>&1
  local rc2=$?
  if [ $rc1 -ne 0 ] || [ $rc2 -ne 0 ]; then
    fail "$1 inference: exit $rc1/$rc2" "$a" "$b"
    return 1
  fi
}

# check_inference <label> <config>
check_inference() {
  local label=$1 a="$TMP/infer_$1_a.out" b="$TMP/infer_$1_b.out" text margin
  infer_twice "$label" "$2" || return
  text=$(generated_text "$a")
  margin=$(sed -nE 's/.*\[Inference\]\[Margin\].*probability: ([0-9.]+) at step.*/\1/p' "$a")
  measure "margin ${label} $margin"
  if [ ${#text} -ne "$MAX_NEW" ]; then
    fail "$label inference: ${#text} generated characters, expected $MAX_NEW" "$a"
  elif [ "$text" != "$(generated_text "$b")" ]; then
    fail "$label inference: two identical runs differ" "$a" "$b"
  elif [ "$text" != "$EXPECTED" ]; then
    fail "$label inference: \"$text\" != expected \"$EXPECTED\" ($(wrong_chars "$text" "$EXPECTED") wrong)" "$a"
  elif ! is_finite "$margin" || ! awk -v m="$margin" -v t="$MIN_MARGIN" 'BEGIN { exit !(m >= t) }'; then
    fail "$label inference: exact, but top-1 margin $margin < $MIN_MARGIN" "$a"
  else
    pass "$label inference: exact continuation, min top-1 margin $margin"
  fi
}

# Tokenizer, dataset, vocabulary size V (sets the loss limit) and steps per
# epoch (set the warmup and the resume step).
WARMUP=0
config base fp32 rope base "$CUDA"
if ! "$BIN" tokenizer_training --config "$TMP/base.yaml" > "$TMP/tok.out" 2>&1 ||
   ! "$BIN" encode --config "$TMP/base.yaml" > "$TMP/encode.out" 2>&1; then
  fail "tokenizer training and encoding" "$TMP/tok.out" "$TMP/encode.out"
  exit 1
fi
"$BIN" train --config "$TMP/base.yaml" --incremental=false --epochs 1 > "$TMP/probe.out" 2>&1
V=$(sed -nE 's/.*model.vocab_size set to ([0-9]+).*/\1/p' "$TMP/probe.out" | head -1)
STEPS_PER_EPOCH=$(final_step "$TMP/probe.out")
if [ -z "$V" ] || [ -z "$STEPS_PER_EPOCH" ]; then
  fail "vocabulary size or steps per epoch not found" "$TMP/probe.out"
  exit 1
fi
LIMIT=$(awk -v v="$V" 'BEGIN { printf "%.4f", 0.5 * log(v) }')
# One epoch of warmup, then the cosine decay.
WARMUP=$STEPS_PER_EPOCH
echo "V=$V, 0.5 ln V = $LIMIT, $STEPS_PER_EPOCH steps per epoch, $EPOCHS epochs"

# CPU backend, in the background while the CUDA runs go.
cpu_pids=()
if [ -n "$CPU" ] && [ -f "$CPU" ]; then
  for pos in rope learned; do
    config "cpu_fp32_${pos}" fp32 "$pos" "cpu_fp32_${pos}" "$CPU"
    "$BIN" train --config "$TMP/cpu_fp32_${pos}.yaml" --incremental=false --epochs $EPOCHS \
      > "$TMP/cpu_fp32_${pos}.out" 2>&1 &
    cpu_pids+=("$!")
  done
fi

declare -A fresh_loss
for prec in fp32 bf16_mixed; do
  for pos in rope learned; do
    case_=${prec}_${pos}
    # Fresh: EPOCHS epochs.
    config "${case_}_fresh" "$prec" "$pos" "${case_}_fresh" "$CUDA"
    out="$TMP/${case_}_fresh.out"
    "$BIN" train --config "$TMP/${case_}_fresh.yaml" --incremental=false --epochs $EPOCHS > "$out" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
      fail "$case_ fresh: exit $rc" "$out"
    elif check_training "$case_ fresh" "$out" $EPOCHS "$out"; then
      fresh_loss[$case_]=$(epoch_field "$out" $EPOCHS train)
      check_inference "${case_}_fresh" "$TMP/${case_}_fresh.yaml"
    fi
    # Resume: the same plan interrupted after EPOCHS-1 epochs, then the last
    # epoch from the checkpoint.
    config "${case_}_resume" "$prec" "$pos" "${case_}_resume" "$CUDA"
    out1="$TMP/${case_}_resume_1.out"
    out2="$TMP/${case_}_resume_2.out"
    "$BIN" train --config "$TMP/${case_}_resume.yaml" --incremental=false --epochs $EPOCHS \
      --stop_after_epoch $((EPOCHS - 1)) > "$out1" 2>&1
    rc1=$?
    "$BIN" train --config "$TMP/${case_}_resume.yaml" --incremental=true --epochs $EPOCHS > "$out2" 2>&1
    rc2=$?
    if [ $rc1 -ne 0 ] || [ $rc2 -ne 0 ]; then
      fail "$case_ resume: exit $rc1/$rc2" "$out1" "$out2"
    elif ! check_steps "$case_ resume" "$out1" "$out2"; then
      :
    elif check_training "$case_ resume" "$out1" $((EPOCHS - 1)) "$out2"; then
      resumed=$(epoch_field "$out2" $EPOCHS train)
      if [ -n "${fresh_loss[$case_]:-}" ]; then
        d=$(absdiff "$resumed" "${fresh_loss[$case_]}")
        measure "resume_delta $case_ $d"
        if within "$resumed" "${fresh_loss[$case_]}" "$TOL_RESUME"; then
          pass "$case_ resume: final train loss $resumed matches fresh ${fresh_loss[$case_]} (|d| $d <= $TOL_RESUME)"
        else
          fail "$case_ resume: final train loss $resumed vs fresh ${fresh_loss[$case_]} (|d| $d > $TOL_RESUME)" "$out2" "$TMP/${case_}_fresh.out"
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
  "$BIN" train --config "$TMP/${name}_a.yaml" --incremental=false --epochs $EPOCHS \
    --stop_after_epoch $((EPOCHS - 1)) > "$TMP/${name}_1.out" 2>&1
  local rc1=$?
  "$BIN" train --config "$TMP/${name}_b.yaml" --incremental=true --epochs $EPOCHS > "$TMP/${name}_2.out" 2>&1
  local rc2=$?
  if [ $rc1 -ne 0 ] || [ $rc2 -ne 0 ]; then
    fail "$from checkpoint resumes under $to (exit $rc1/$rc2)" "$TMP/${name}_1.out" "$TMP/${name}_2.out"
  elif check_steps "$from checkpoint resumed under $to" "$TMP/${name}_1.out" "$TMP/${name}_2.out"; then
    check_training "$from checkpoint resumed under $to" \
      "$TMP/${name}_1.out" $((EPOCHS - 1)) "$TMP/${name}_2.out"
  fi
  config "${name}_infer" "$to" rope "${from}_rope_fresh" "$CUDA"
  check_inference "${from}_checkpoint_under_$to" "$TMP/${name}_infer.yaml"
}
cross fp32 bf16_mixed
cross bf16_mixed fp32

# Negative control: an untrained checkpoint must miss the expected text.
config untrained fp32 rope untrained "$CUDA" 0.000000001
"$BIN" train --config "$TMP/untrained.yaml" --incremental=false --epochs 1 > "$TMP/untrained.out" 2>&1
if [ $? -ne 0 ]; then
  fail "negative control: training at learning rate 1e-9 failed" "$TMP/untrained.out"
elif infer_twice untrained "$TMP/untrained.yaml"; then
  text=$(generated_text "$TMP/infer_untrained_a.out")
  wrong=$(wrong_chars "$text" "$EXPECTED")
  measure "untrained_wrong $wrong"
  if [ ${#text} -eq "$MAX_NEW" ] && [ "$wrong" -ge "$NEG_MIN_WRONG" ]; then
    pass "negative control: untrained checkpoint misses $wrong of $MAX_NEW characters"
  else
    fail "negative control: untrained checkpoint misses only $wrong of $MAX_NEW characters" "$TMP/infer_untrained_a.out"
  fi
fi

# CPU backend: collect the background runs.
if [ ${#cpu_pids[@]} -gt 0 ]; then
  i=0
  for pos in rope learned; do
    wait "${cpu_pids[$i]}"
    rc=$?
    i=$((i + 1))
    out="$TMP/cpu_fp32_${pos}.out"
    if [ $rc -ne 0 ]; then
      fail "CPU fp32 $pos: exit $rc" "$out"
    elif check_training "CPU fp32 $pos" "$out" $EPOCHS "$out"; then
      cpu=$(epoch_field "$out" $EPOCHS train)
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
      check_inference "cpu_fp32_${pos}" "$TMP/cpu_fp32_${pos}.yaml"
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
