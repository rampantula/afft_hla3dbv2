#!/usr/bin/env python3

import os
import shutil

"""
Copyright (c) 2026 The Children's Hospital of Philadelphia and Stanford University
Licensed for academic and non-commercial use only. Commercial use requires a separate license.
See LICENSE file for details.
"""

INPUT_ROOT = "outfiles"
OUTPUT_ROOT = "MHC_pdbs"


def main():
    input_root = os.path.abspath(INPUT_ROOT)
    output_root = os.path.abspath(OUTPUT_ROOT)

    os.makedirs(output_root, exist_ok=True)

    if not os.path.isdir(input_root):
        raise FileNotFoundError(f"Input directory does not exist: {input_root}")

    for target in sorted(os.listdir(input_root)):
        target_dir = os.path.join(input_root, target)

        if not os.path.isdir(target_dir):
            continue

        target_output_dir = os.path.join(target_dir, "outputs")
        src_pdb = os.path.join(
            target_output_dir,
            f"{target}_model_split.pdb",
        )
        dst_pdb = os.path.join(output_root, f"{target}.pdb")

        if not os.path.isfile(src_pdb):
            print(f"[!] expected PDB not found: {src_pdb}")
            continue

        try:
            shutil.copy(src_pdb, dst_pdb)
            print(f"[+] moved {src_pdb!r} -> {dst_pdb!r}")
        except Exception as exc:
            print(f"[!] failed to move {src_pdb!r}: {exc}")


if __name__ == "__main__":
    main()