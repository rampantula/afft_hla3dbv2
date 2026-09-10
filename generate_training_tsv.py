#!/usr/bin/env python3
"""
Generate an AlphaFold fine-tuning training_dataset.tsv directly from an Excel file.

Expected Excel columns:
  - PDB ID
  - MHC_sequence
  - Peptide Sequence

No target PDB directory is required.

Output columns:
  targetid
  target_chainseq
  templates_alignfile
  native_pdbfile
  native_alignstring
  native_exists
  native_identities
  native_len
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def normalize_pdb_id(value) -> str:
    """Convert spreadsheet cell value to a clean target identifier."""
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text


def normalize_sequence(value) -> str:
    """Normalize an amino-acid sequence from Excel."""
    if pd.isna(value):
        return ""
    return "".join(str(value).split()).upper()


def build_training_tsv(
    excel_file: Path,
    output_tsv: Path,
    alignments_dir: Path,
    pdb_id_column: str = "PDB ID",
    mhc_sequence_column: str = "MHC_sequence",
    peptide_sequence_column: str = "Peptide Sequence",
) -> pd.DataFrame:
    df = pd.read_excel(excel_file, dtype=str)

    required_columns = [
        pdb_id_column,
        mhc_sequence_column,
        peptide_sequence_column,
    ]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise KeyError(
            f"Missing required Excel column(s): {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    rows = []

    for excel_row_number, (_, record) in enumerate(df.iterrows(), start=2):
        pdb_id = normalize_pdb_id(record[pdb_id_column])
        if not pdb_id:
            continue

        mhc_seq = normalize_sequence(record[mhc_sequence_column])
        pep_seq = normalize_sequence(record[peptide_sequence_column])

        if not mhc_seq:
            raise ValueError(
                f"Missing MHC sequence for target {pdb_id!r} "
                f"(Excel row {excel_row_number})"
            )
        if not pep_seq:
            raise ValueError(
                f"Missing peptide sequence for target {pdb_id!r} "
                f"(Excel row {excel_row_number})"
            )

        target_chainseq = f"{mhc_seq}/{pep_seq}"

        rows.append({
            "targetid": pdb_id,
            "target_chainseq": target_chainseq,
            "templates_alignfile": str(
                alignments_dir / f"{pdb_id}__alignments.tsv"
            ),

            # No native target PDB is supplied in this workflow.
            "native_pdbfile": "",
            "native_alignstring": "",
            "native_exists": "FALSE",
            "native_identities": "",
            "native_len": "",
        })

    out_df = pd.DataFrame(
        rows,
        columns=[
            "targetid",
            "target_chainseq",
            "templates_alignfile",
            "native_pdbfile",
            "native_alignstring",
            "native_exists",
            "native_identities",
            "native_len",
        ],
    )

    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_tsv, sep="\t", index=False)
    return out_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate training_dataset.tsv from Excel MHC and peptide sequences "
            "without requiring target PDB structures."
        )
    )

    parser.add_argument(
        "--excel",
        required=True,
        type=Path,
        help="Input .xlsx file containing PDB ID, MHC_sequence, and Peptide Sequence",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output training .tsv",
    )
    parser.add_argument(
        "--alignments-dir",
        required=True,
        type=Path,
        help="Folder/path prefix used in templates_alignfile values",
    )
    parser.add_argument(
        "--pdb-id-column",
        default="PDB ID",
        help="Excel column containing target/PDB IDs. Default: 'PDB ID'",
    )
    parser.add_argument(
        "--mhc-sequence-column",
        default="MHC_sequence",
        help="Excel column containing MHC amino-acid sequences. Default: 'MHC_sequence'",
    )
    parser.add_argument(
        "--peptide-sequence-column",
        default="Peptide Sequence",
        help="Excel column containing peptide sequences. Default: 'Peptide Sequence'",
    )

    args = parser.parse_args()

    out_df = build_training_tsv(
        excel_file=args.excel,
        output_tsv=args.out,
        alignments_dir=args.alignments_dir,
        pdb_id_column=args.pdb_id_column,
        mhc_sequence_column=args.mhc_sequence_column,
        peptide_sequence_column=args.peptide_sequence_column,
    )

    print(f"Wrote {len(out_df)} rows to {args.out}")


if __name__ == "__main__":
    main()
