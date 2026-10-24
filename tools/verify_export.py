#!/usr/bin/env python3
"""Load a `litnicelm export` model.safetensors with the official safetensors
reader and print its metadata and every tensor's name, shape and dtype, plus
config.json. A manual check of the format (needs tools/requirements.txt).

Example:
  tools/verify_export.py hf_out
"""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("export_dir", type=Path, help="output of `litnicelm export --out`")
    args = parser.parse_args()

    from safetensors import safe_open

    path = args.export_dir / "model.safetensors"
    if not path.is_file():
        sys.exit(f"{path}: not found")
    total = 0
    with safe_open(str(path), framework="numpy") as f:
        print(f"metadata: {f.metadata()}")
        for name in f.keys():
            tensor = f.get_tensor(name)
            total += tensor.size
            print(f"  {name:40s} {str(list(tensor.shape)):16s} {tensor.dtype}")
    print(f"{total} parameters")
    config = json.loads((args.export_dir / "config.json").read_text())
    print("config.json:", json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
