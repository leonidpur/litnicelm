# Fast Try: CUDA/cuBLAS

Use this path on a machine with an NVIDIA GPU and CUDA Toolkit. It creates only
local disposable files under ignored directories.

## 1. Install dependencies

Ubuntu/Debian common dependencies:

```bash
sudo apt install cmake g++ pkg-config libsentencepiece-dev
```

Install the NVIDIA CUDA Toolkit separately so `nvcc`, CUDA runtime, and cuBLAS
are visible to CMake.

The script checks for common dependencies and `nvcc`. It does not run `sudo apt
install` for you.

## 2. Run the CUDA/cuBLAS try script

From the repository root:

```bash
./scripts/try_cuda.sh
```

The script will:

- create `data/local/abc.txt`
- create `conf/local/try_cuda.yaml`
- build the project
- verify `build/liblitnice_backend_cublas_plugin.so` exists
- train tokenizer artifacts
- tokenize the tiny corpus
- train one short CUDA/cuBLAS epoch
- run inference with the prompt `a b c `

## 3. Expected result

You should see build output, tokenizer/tokenization logs, one short training
run, and then generated text. The text is not expected to be useful; this is a
smoke test proving that the executable, CUDA/cuBLAS backend plugin, config
loading, tokenization, training, checkpointing, and inference path all work.

Generated files are local and ignored by git:

```text
data/local/
artifacts/local/
checkpoints/local/
journal/local/
conf/local/
```
