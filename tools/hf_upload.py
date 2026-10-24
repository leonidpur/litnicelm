#!/usr/bin/env python3
"""Upload a `litnicelm export` directory to a Hugging Face model repo.

The directory (model.safetensors, config.json, tokenizer/, README.md) goes to
the repo root; with --tb-dir, the run's TensorBoard logs go under runs/.
Authentication: HF_TOKEN, `huggingface-cli login`, or --token.

Examples:
  tools/hf_upload.py hf_out --repo user/litnicelm-doyle --dry-run
  tools/hf_upload.py hf_out --repo user/litnicelm-doyle --tb-dir tensorboard/<run_id>
  tools/hf_upload.py hf_out --repo user/litnicelm-doyle --private
"""

import argparse
import sys
from pathlib import Path

REQUIRED = ["model.safetensors", "config.json", "README.md"]


def files_under(root):
    return sorted(p for p in root.rglob("*") if p.is_file())


def plan(export_dir, tb_dir):
    """(local path, path in repo) for every file to upload."""
    items = [(p, p.relative_to(export_dir).as_posix()) for p in files_under(export_dir)]
    if tb_dir is not None:
        items += [(p, "runs/" + p.relative_to(tb_dir).as_posix()) for p in files_under(tb_dir)]
    return items


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("export_dir", type=Path, help="output of `litnicelm export --out`")
    parser.add_argument("--repo", required=True, help="model repo id, e.g. user/name")
    parser.add_argument("--tb-dir", type=Path, help="TensorBoard log directory of the run")
    parser.add_argument("--private", action="store_true", help="create the repo as private")
    parser.add_argument("--token", help="Hugging Face token (default: HF_TOKEN or login)")
    parser.add_argument("--message", default="Upload LitniceLM export",
                        help="commit message")
    parser.add_argument("--dry-run", action="store_true",
                        help="list what would be uploaded; no network access")
    args = parser.parse_args()

    missing = [f for f in REQUIRED if not (args.export_dir / f).is_file()]
    if missing:
        sys.exit(f"{args.export_dir}: missing {', '.join(missing)} (run `litnicelm export` first)")
    if args.tb_dir is not None and not args.tb_dir.is_dir():
        sys.exit(f"--tb-dir {args.tb_dir}: not a directory")

    items = plan(args.export_dir, args.tb_dir)
    total = sum(p.stat().st_size for p, _ in items)
    print(f"{'private' if args.private else 'public'} repo {args.repo}: "
          f"{len(items)} files, {total / 2**20:.2f} MiB")
    for local, remote in items:
        print(f"  {remote}  ({local.stat().st_size} bytes)")
    if args.dry_run:
        return

    from huggingface_hub import HfApi  # imported here so --dry-run needs no packages

    api = HfApi(token=args.token)
    url = api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
    api.upload_folder(repo_id=args.repo, repo_type="model", folder_path=str(args.export_dir),
                      commit_message=args.message)
    if args.tb_dir is not None:
        api.upload_folder(repo_id=args.repo, repo_type="model", folder_path=str(args.tb_dir),
                          path_in_repo="runs", commit_message=args.message + " (TensorBoard logs)")
    print(f"uploaded to {url}")


if __name__ == "__main__":
    main()
