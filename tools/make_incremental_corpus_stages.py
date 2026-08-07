#!/usr/bin/env python3
import argparse
import re
from pathlib import Path


def replace_yaml_scalar(text, key_path, value):
    indent_key = key_path.split(".")[-1]
    pattern = re.compile(rf"^(\s*{re.escape(indent_key)}:\s*).*$", re.MULTILINE)
    replacement = rf'\1"{value}"'
    text, count = pattern.subn(replacement, text, count=1)
    if count != 1:
        raise RuntimeError(f"failed to replace YAML key: {key_path}")
    return text


def replace_yaml_bool(text, key_path, value):
    indent_key = key_path.split(".")[-1]
    pattern = re.compile(rf"^(\s*{re.escape(indent_key)}:\s*).*$", re.MULTILINE)
    replacement = rf"\1{'true' if value else 'false'}"
    text, count = pattern.subn(replacement, text, count=1)
    if count != 1:
        raise RuntimeError(f"failed to replace YAML key: {key_path}")
    return text


def token_prefixes(source_text, start_tokens, step_tokens, max_tokens):
    matches = list(re.finditer(r"\S+", source_text))
    if not matches:
        raise RuntimeError("source corpus has no tokens")

    total_tokens = len(matches)
    limit = min(max_tokens or total_tokens, total_tokens)
    sizes = list(range(start_tokens, limit + 1, step_tokens))
    if sizes[-1] != limit:
        sizes.append(limit)

    for size in sizes:
      end = matches[size - 1].end()
      yield size, source_text[:end].rstrip() + "\n"


def main():
    parser = argparse.ArgumentParser(
        description="Generate incremental corpus/config stages from a source text.")
    parser.add_argument("--source", required=True, help="Source text file.")
    parser.add_argument("--template", required=True, help="Template YAML config.")
    parser.add_argument("--output-root", required=True,
                        help="Directory for generated corpora/artifacts/datasets/checkpoints.")
    parser.add_argument("--config-dir", required=True,
                        help="Directory for generated YAML configs.")
    parser.add_argument("--name", default="syntetic1",
                        help="Stage name prefix.")
    parser.add_argument("--start-tokens", type=int, default=16)
    parser.add_argument("--step-tokens", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=0,
                        help="0 means all source tokens.")
    args = parser.parse_args()

    source_path = Path(args.source).resolve()
    template_path = Path(args.template).resolve()
    output_root = Path(args.output_root).resolve()
    config_dir = Path(args.config_dir).resolve()

    if args.start_tokens <= 0 or args.step_tokens <= 0:
        raise RuntimeError("start-tokens and step-tokens must be positive")

    source_text = source_path.read_text(encoding="utf-8")
    template = template_path.read_text(encoding="utf-8")

    corpus_dir = output_root / "corpus"
    artifacts_root = output_root / "artifacts"
    tokenized_dir = output_root / "tokenized"
    checkpoint_dir = output_root / "checkpoints"
    journal_dir = output_root / "journals"
    for directory in [corpus_dir, artifacts_root, tokenized_dir,
                      checkpoint_dir, journal_dir, config_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    manifest_lines = [
        "stage\ttokens\tcorpus\tconfig\ttokenizer_cmd\ttokenize_cmd\ttrain_cmd"
    ]

    for stage_idx, (token_count, corpus_text) in enumerate(
            token_prefixes(source_text, args.start_tokens, args.step_tokens,
                           args.max_tokens),
            start=1):
        stage = f"{args.name}_stage_{stage_idx:03d}_t{token_count:04d}"
        corpus_path = corpus_dir / f"{stage}.txt"
        config_path = config_dir / f"{stage}.yaml"
        artifacts_dir = artifacts_root / stage
        tokenized_path = tokenized_dir / f"{stage}.bin"
        latest_path = checkpoint_dir / f"{stage}_latest.ckpt"
        best_path = checkpoint_dir / f"{stage}_best.ckpt"
        journal_path = journal_dir / f"{stage}.txt"

        corpus_path.write_text(corpus_text, encoding="utf-8")

        cfg = template
        cfg = replace_yaml_scalar(cfg, "tokenizer.training_corpus", str(corpus_path))
        cfg = replace_yaml_scalar(cfg, "tokenizer.artifacts_dir", str(artifacts_dir))
        cfg = replace_yaml_scalar(cfg, "tokenization.input_corpus", str(corpus_path))
        cfg = replace_yaml_scalar(cfg, "tokenization.output_binary", str(tokenized_path))
        cfg = replace_yaml_bool(cfg, "training.incremental", False)
        cfg = replace_yaml_scalar(cfg, "paths.model_file_latest", str(latest_path))
        cfg = replace_yaml_scalar(cfg, "paths.model_file_best", str(best_path))
        cfg = replace_yaml_scalar(cfg, "paths.journal_file", str(journal_path))
        config_path.write_text(cfg, encoding="utf-8")

        manifest_lines.append(
            "\t".join([
                stage,
                str(token_count),
                str(corpus_path),
                str(config_path),
                f"./build/litnicelm --tokenizer_training --config {config_path}",
                f"./build/litnicelm --tokenize --config {config_path}",
                f"./build/litnicelm --train --config {config_path}",
            ]))

    manifest_path = output_root / "manifest.tsv"
    manifest_path.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    print(f"wrote stages manifest: {manifest_path}")


if __name__ == "__main__":
    main()
