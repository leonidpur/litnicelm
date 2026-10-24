# Working rules

The owner reviews architecture, not every line. These rules apply to all
work in this repository.

## Scope

- Make exactly the change that was asked for: no related cleanups, extra
  files or restructuring. If a larger or different change seems better,
  describe it and ask first.
- A question gets an answer (read-only investigation is fine) and an offer;
  no edits, builds or commits.
- No drive-by refactors. Note them in `BACKLOG.md` instead.

## Features

- Every feature starts with a design note (about 10 lines): modules
  touched, new classes and interfaces, changes to existing interfaces, new
  dependencies, data flow. Wait for approval; after it, implement without
  further questions.
- Prefer new classes implementing existing interfaces over modifying
  existing code.
- Top-level components (`Trainer::train`, entry points, controllers) read
  as orchestration, one call per concept. Mechanics go into a dedicated
  class or a well-named method.

## Commits

- Commit only after explicit approval of that commit (approving a plan is
  not approval to commit). The same for push.
- First commit of a feature: interfaces and declarations only. Then
  implementation, wiring (config, CLI), tests, as separate commits.
- Soft limit of about 300 lines of non-test code per commit; split above.
- Never mix mechanical changes (renames, moves, formatting) with logic.
- Each commit report lists new types, changed interfaces, files touched and
  `git diff --stat`. Changes to an existing interface or the plugin ABI are
  called out explicitly.

## Data

- Tokenized datasets (`*.bin`) and checkpoints (`*.ckpt` with their `.run`
  and `.tokfp` sidecars) live on the external SSD under
  `/mnt/ext_ssd/litnicelm/<project>/`, never in the repository tree.

## Mixed-precision kernels

- No templates in device code: one concrete `__global__` per variant,
  `<op>_f32` and `<op>_bf16act`, free to diverge.
- `_bf16act`: activations and activation gradients BF16; parameters,
  parameter gradients and optimizer state FP32. The variant is chosen on
  the host from the operand dtypes.
- No load/store helpers in kernels: index typed pointers directly and
  convert with the CUDA intrinsics.
- Each kernel pair is tested on identical BF16-representable inputs.
