# Fast Try: CPU

Use this path on a modest machine with no GPU. It creates only local disposable
files under ignored directories.

## 1. Install dependencies

Ubuntu/Debian:

```bash
sudo apt install cmake g++ pkg-config libsentencepiece-dev
```

The script checks for these dependencies and prints this command if something is
missing. It does not run `sudo apt install` for you.

## 2. Run the CPU try script

From the repository root:

```bash
./scripts/try_cpu.sh
```

The script will:

- create `data/local/abc.txt`
- create `conf/local/try_cpu.yaml`
- build the project
- train tokenizer artifacts
- tokenize the tiny corpus
- train one short epoch
- run inference with the prompt `a b c `

## 3. Expected result

You should see build output, tokenizer/tokenization logs, one short training
run, and then generated text. The text is not expected to be useful; this is a
smoke test proving that the executable, CPU backend, config loading,
tokenization, training, checkpointing, and inference path all work.

Generated files are local and ignored by git:

```text
data/local/
artifacts/local/
checkpoints/local/
journal/local/
conf/local/
```
