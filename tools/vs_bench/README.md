# LitniceLM comparison

This directory is a standalone comparison project. The runner first explains
the complete plan, then executes parity and isolated training-speed runs.

Run it from the `tools/` directory:

```bash
python3 -m vs_bench
```

The comparison is configured by the `SPEC` instance at the top of `run.py`.
There is intentionally no command-line interface.

The runner locks the GPU at 1500 MHz with `sudo -n /usr/bin/nvidia-smi`,
verifies the sampled clock under load, and resets it on exit. It never opens a
password prompt. If permission is unavailable, it stops before training and
explains the narrowly scoped sudoers rule or manual lock command.

Child-process logs are written outside the repository under
`/mnt/ext_ssd/litnicelm/bench/<date>/`. Compact structured results,
environment metadata, and PyTorch worker request JSON files are written there
as well. Each green PyTorch command names its request file and can be rerun
directly. The runner does not overwrite files under `docs/`.

Whole-process timing is differential: LitniceLM runs one and three complete
epochs with validation and checkpointing disabled, and PyTorch runs the exact
optimizer-step counts observed at those endpoints. Startup and teardown cost
cancel in `(T3 - T1) / (steps3 - steps1)`.

Every matched-step row also reports the realized layer/head/hidden/FFN
dimensions, tokens per optimizer step, and final training loss. The run fails
if dimensions or token counts differ, or if PyTorch's final loss is more than
the configured tolerance from LitniceLM's loss for the same precision and
repeat. PyTorch warms its CUDA kernels first, then resets model weights,
optimizer state, and the data cursor so warm-up updates do not enter the
matched-step loss.

