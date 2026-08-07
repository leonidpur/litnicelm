# litnicelm

`litnicelm` is a small C++17 language-model training and inference project. It
builds a command-line executable plus pluggable tensor backends, then drives
tokenizer training, corpus tokenization, model training, inspection, and text
generation from YAML configuration files.

The code is intentionally low-level: model memory layout, backend dispatch,
tokenization, checkpoints, journals, and training diagnostics are all visible in
the repository.

## Features

- CMake-based C++17 build.
- Character and SentencePiece tokenizer support.
- CPU backend built by default.
- Optional OpenBLAS backend plugin.
- Optional CUDA/cuBLAS backend plugin.
- YAML-driven model, tokenizer, backend, training, and inference settings.
- Training, dry-run, tokenizer, tokenization, inference, and next-token
  inspection modes.

## Requirements

Required:

- CMake 3.16 or newer
- A C++17 compiler
- `pkg-config`
- SentencePiece development package

Optional:

- OpenBLAS development package for `liblitnice_backend_openblas_plugin.so`
- CUDA Toolkit for `liblitnice_backend_cublas_plugin.so`

On Ubuntu/Debian:

```bash
sudo apt install cmake g++ pkg-config libsentencepiece-dev
sudo apt install libopenblas-dev   # optional
```

Install the CUDA Toolkit separately if you want the CUDA/cuBLAS backend.

## Build

From the repository root:

```bash
cmake -S . -B build
cmake --build build -j
```

The main executable is:

```bash
./build/litnicelm
```

The CPU backend is built as:

```bash
./build/liblitnice_backend_cpu.so
```

OpenBLAS and CUDA plugins are built only when their dependencies are available.
CMake prints a status message when it skips one of them.

## Quickstart

The repository contains configs, but it does not currently include the corpora
referenced by those configs. For a tiny CPU smoke test, create the sample corpus
used by `conf/char_abc.yaml`:

```bash
mkdir -p data checkpoints journal
printf 'abcabcabcabcabcabcabcabc\n' > data/abc.txt
```

Build:

```bash
cmake -S . -B build
cmake --build build -j
```

Train tokenizer artifacts:

```bash
./build/litnicelm tokenizer_training --config conf/char_abc.yaml
```

Tokenize the corpus:

```bash
./build/litnicelm encode --config conf/char_abc.yaml
```

Run a short training pass:

```bash
./build/litnicelm train --config conf/char_abc.yaml --epochs 1
```

Generate text:

```bash
./build/litnicelm infer --config conf/char_abc.yaml --prompt "ab" --max_new 32
```

## CLI

Show help:

```bash
./build/litnicelm --help
```

Train:

```bash
./build/litnicelm train --config <config.yaml> [train_seq_len] [batch_size]
```

Useful train flags:

```bash
--epochs N
--incremental
--no-incremental
--epoch_report_every N
--probe embeddings,output_head,loss,backward,attention,ffn,layernorm
--do-probe
--logit
```

Dry run:

```bash
./build/litnicelm dry_run --config <config.yaml>
```

Train tokenizer artifacts:

```bash
./build/litnicelm tokenizer_training --config <config.yaml>
```

Tokenize a corpus:

```bash
./build/litnicelm encode --config <config.yaml>
```

Inference:

```bash
./build/litnicelm infer --config <config.yaml> --prompt "text" --max_new 128
```

Inference flags:

```bash
--prompt TEXT
--max_new N
--temp X
--top_k K
--top_p P
--seed N
```

Inspect the next-token distribution:

```bash
./build/litnicelm inspect --config <config.yaml> --prompt "text"
```

Interactive inference loop:

```bash
./build/litnicelm inferloop --config <config.yaml>
```

## Configuration

Configs live under `conf/`. The main sections are:

- `backend.library`: shared backend library to load.
- `tokenizer`: tokenizer type, training corpus, artifact directory, and
  validation options.
- `tokenization`: input corpus and output tokenized dataset path.
- `model`: sequence length, layer count, head count, hidden size, and FFN size.
- `training`: optimizer, epoch, batch, stride, checkpoint, and early-stop
  settings.
- `paths`: checkpoint and journal output paths.
- `inference`: default prompt, sampling settings, and seed.
- `logging` and `reporting`: console and diagnostic verbosity.

You can override selected config values from the CLI, and environment overrides
are applied when `ENV_PREFIX` is set.

## Backends

Most configs point at one of these libraries:

```yaml
backend:
  library: "./build/liblitnice_backend_cpu.so"
```

```yaml
backend:
  library: "./build/liblitnice_backend_openblas_plugin.so"
```

```yaml
backend:
  library: "./build/liblitnice_backend_cublas_plugin.so"
```

Use the CPU backend for the simplest local smoke test. Use OpenBLAS or CUDA when
the corresponding plugin was built and the config points at the right `.so`.

## Workflow Targets

CMake also defines convenience targets that run the executable with
`WORKFLOW_CONFIG`:

```bash
cmake --build build --target tokenizer_training
cmake --build build --target encode
cmake --build build --target train
cmake --build build --target dry_run
cmake --build build --target infer
```

Override the workflow config at configure time:

```bash
cmake -S . -B build -DWORKFLOW_CONFIG="$PWD/conf/char_abc.yaml"
```

For inference, also set:

```bash
cmake -S . -B build \
  -DWORKFLOW_CONFIG="$PWD/conf/char_abc.yaml" \
  -DWORKFLOW_INFER_PROMPT="ab"
```

## Repository Layout

```text
backends/                         Backend interface headers and GPU op code
conf/                             Example YAML configs
include/                          Public headers
src/backend_libs/                 OpenBLAS and CUDA backend plugins
src/config/                       YAML mapping and validation
src/model/                        Tensor, memory, model, trainer, inference
src/observers/                    Training journals, checkpoints, reporting
src/tokenizer/                    Tokenizer plugins and corpus tokenization
src/utils/                        CLI parser and command executor
tools/                            Helper scripts for staged experiments
```

## Notes

- Paths inside configs are resolved relative to the current working directory.
  Run commands from the repository root unless you intentionally use absolute
  paths.
- Many example configs reference local datasets under `/mnt/ext_ssd/litnicelm`.
  Adjust those paths before running them on another machine.
- Checkpoints and journals are created as configured under `paths`.
