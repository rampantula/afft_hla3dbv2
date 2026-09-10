#!/usr/bin/env python3
"""
Copyright (c) 2026 The Children's Hospital of Philadelphia and Stanford University
Licensed for academic and non-commercial use only. Commercial use requires a separate license.
See LICENSE file for details.
"""

from __future__ import annotations

import argparse
import glob
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

try:
    from Bio.Align import PairwiseAligner, substitution_matrices
    BLOSUM62 = substitution_matrices.load("BLOSUM62")
except Exception as exc:
    raise SystemExit("Biopython is required: pip install biopython") from exc

# Configured once and reused for every MHC alignment. PairwiseAligner is the
# C-accelerated replacement for the (deprecated, pure-Python) pairwise2 --
# ~24x faster on MHC-length sequences with identical global-alignment scores
# and residue mappings for the BLOSUM62 / gap-open -11 / gap-extend -1 scheme
# used below.
_MHC_ALIGNER = PairwiseAligner()
_MHC_ALIGNER.substitution_matrix = BLOSUM62
_MHC_ALIGNER.open_gap_score = -11
_MHC_ALIGNER.extend_gap_score = -1
_MHC_ALIGNER.mode = "global"


AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "MSE": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y",
    "VAL": "V",
}


# --------------------------------------------------------------------------
# Template PDB parsing (same approach as the Excel/PDB version of this
# script: sequences are read directly from the structure, not from a
# separate sequence file)
# --------------------------------------------------------------------------

def read_pdb_chain_sequences(pdb_path: Path) -> Dict[str, str]:
    """Read one-letter sequences from ATOM/HETATM records, preserving residue order."""
    seqs: Dict[str, List[str]] = {}
    seen = set()

    with open(pdb_path, "r") as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            if line[17:20].strip() == "HOH":
                continue
            altloc = line[16]
            if altloc not in " A1":
                continue
            res3 = line[17:20].strip()
            if res3 not in AA3_TO_1:
                continue

            chain = line[21].strip() or "_"
            resid = line[22:27]
            key = (chain, resid)
            if key in seen:
                continue
            seen.add(key)
            seqs.setdefault(chain, []).append(AA3_TO_1[res3])

    return {chain: "".join(seq) for chain, seq in seqs.items()}


def pdb_id_from_template_path(path: Path) -> str:
    """Normalize template stem so self-template filtering works with *_reordered.pdb."""
    stem = path.stem
    if stem.endswith("_reordered"):
        stem = stem[: -len("_reordered")]
    return stem


# --------------------------------------------------------------------------
# Text sequence I/O (same two-line format as initialize.py's get_seq)
# --------------------------------------------------------------------------

def get_seq(filename: Path) -> Tuple[str, str]:
    """Read (hla_seq, pep_seq) from a two-line sequence text file."""
    with open(filename, "r") as file:
        lines = file.readlines()
    if len(lines) < 2:
        raise ValueError(f"Expected 2 lines (hla_seq, pep_seq) in {filename}, got {len(lines)}")
    hla_seq = lines[0].strip()
    pep_seq = lines[1].strip()
    if not hla_seq or not pep_seq:
        raise ValueError(f"Empty hla_seq or pep_seq in {filename}")
    return hla_seq, pep_seq


def pdbid_from_seq_filename(seq_file: Path) -> str:
    """Same convention as initialize.py: text before the first underscore."""
    return seq_file.name.split("_")[0]


# --------------------------------------------------------------------------
# Scoring (unchanged logic)
# --------------------------------------------------------------------------

def peptide_mismatches(p1: str, p2: str) -> int:
    """Hamming distance for 9-mer peptides."""
    if len(p1) != 9 or len(p2) != 9:
        raise ValueError(f"Expected both peptides to be 9-mers, got {len(p1)} and {len(p2)}")
    return sum(a != b for a, b in zip(p1, p2))


def peptide_blosum_score(p1: str, p2: str) -> float:
    """Sum BLOSUM62 substitution scores over aligned 9-mer positions."""
    if len(p1) != 9 or len(p2) != 9:
        raise ValueError(f"Expected both peptides to be 9-mers, got {len(p1)} and {len(p2)}")

    score = 0.0
    for a, b in zip(p1, p2):
        try:
            score += float(BLOSUM62[a, b])
        except Exception:
            score += float(BLOSUM62[b, a])
    return score


def needleman_wunsch_mhc(target_mhc: str, template_mhc: str):
    """Return (aligned_target, aligned_template, score) for the best global
    MHC alignment, using the shared PairwiseAligner."""
    aln = _MHC_ALIGNER.align(target_mhc, template_mhc)[0]
    # aln[0]/aln[1] are the gapped target/template strings (with '-').
    return str(aln[0]), str(aln[1]), float(aln.score)


def mhc_alignment_mapping_identity_score(
    target_mhc: str,
    template_mhc: str,
) -> Tuple[Dict[int, int], float, float]:
    """Return target-to-template MHC mapping, MHC identity, and MHC alignment score."""
    aligned_target, aligned_template, score = needleman_wunsch_mhc(target_mhc, template_mhc)

    i = j = 0
    mapping: Dict[int, int] = {}
    matches = 0
    aligned_pairs = 0

    for a, b in zip(aligned_target, aligned_template):
        has_a = a != "-"
        has_b = b != "-"

        if has_a and has_b:
            mapping[i] = j
            aligned_pairs += 1
            if a == b:
                matches += 1

        if has_a:
            i += 1
        if has_b:
            j += 1

    identity = matches / aligned_pairs if aligned_pairs else 0.0
    return mapping, identity, score


def peptide_position_mapping(
    target_mhc_len: int,
    template_mhc_len: int,
    peptide_len: int = 9,
) -> Dict[int, int]:
    """For 9-mer Class I peptides, map peptide residues positionally one-to-one."""
    return {
        target_mhc_len + k: template_mhc_len + k
        for k in range(peptide_len)
    }


def mapping_to_alignstring(mapping: Dict[int, int]) -> str:
    return ";".join(f"{i}:{j}" for i, j in sorted(mapping.items()))


def identity_count(target_seq: str, template_seq: str, mapping: Dict[int, int]) -> int:
    return sum(
        1
        for i, j in mapping.items()
        if i < len(target_seq) and j < len(template_seq) and target_seq[i] == template_seq[j]
    )


def template_pdb_group(template_id: str) -> str:
    """Return the base PDB group used to prevent duplicate template complexes."""
    return template_id.split("-", 1)[0].upper()


def minmax_normalize(values: List[float]) -> List[float]:
    """Min-max normalize values. If all values are equal, return 1.0 for all candidates."""
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi == lo:
        return [1.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


# --------------------------------------------------------------------------
# Per-target template selection (unchanged scoring/filter/selection logic)
# --------------------------------------------------------------------------

def select_templates_for_target(
    target_pdbid: str,
    target_mhc: str,
    target_pep: str,
    template_records: List[dict],
    args: argparse.Namespace,
    mhc_weight: float,
    peptide_weight: float,
    target_dir: Path,
):
    target_full = target_mhc + target_pep
    candidates = []

    for rec in template_records:
        if rec["id"] == target_pdbid:
            continue

        mismatches = peptide_mismatches(target_pep, rec["pep"])
        if not args.disable_peptide_mismatch_filter:
            if mismatches < args.min_peptide_mismatches:
                continue

        mhc_mapping, mhc_identity, mhc_score = mhc_alignment_mapping_identity_score(target_mhc, rec["mhc"])
        if not args.disable_mhc_identity_filter:
            if mhc_identity > args.max_mhc_identity:
                continue

        peptide_score = peptide_blosum_score(target_pep, rec["pep"])

        full_mapping = dict(mhc_mapping)
        full_mapping.update(peptide_position_mapping(len(target_mhc), len(rec["mhc"]), peptide_len=9))
        alignstring = mapping_to_alignstring(full_mapping)

        # Destination path after copying, matching initialize.py's layout:
        # <output_root>/<target_pdbid>/inputs/templates/<template_pdbid_lower>.pdb
        dest_template_pdbfile = str(target_dir / "templates" / f"{rec['id'].lower()}.pdb")

        candidates.append({
            "template_pdbfile": dest_template_pdbfile,
            "target_to_template_alignstring": alignstring,
            "identities": identity_count(target_full, rec["full"], full_mapping),
            "target_len": len(target_full),
            "template_len": len(rec["full"]),
            "_template_id": rec["id"],
            "_pdb_group": rec["pdb_group"],
            "_source_pdb_path": rec["source_pdb_path"],
            "_mhc_score": mhc_score,
            "_mhc_identity": mhc_identity,
            "_peptide_score": peptide_score,
            "_peptide_mismatches": mismatches,
        })

    if not candidates:
        return [], []

    norm_mhc_scores = minmax_normalize([row["_mhc_score"] for row in candidates])
    norm_peptide_scores = minmax_normalize([row["_peptide_score"] for row in candidates])

    for row, norm_mhc, norm_pep in zip(candidates, norm_mhc_scores, norm_peptide_scores):
        row["_norm_mhc_score"] = norm_mhc
        row["_norm_peptide_score"] = norm_pep
        row["_combined_score"] = mhc_weight * norm_mhc + peptide_weight * norm_pep

    candidates.sort(
        key=lambda x: (
            x["_combined_score"],
            x["_norm_mhc_score"],
            x["_norm_peptide_score"],
            x["_peptide_mismatches"],
        ),
        reverse=True,
    )

    if args.allow_same_pdb_multiple_times:
        selected = candidates[: args.top_n]
    else:
        selected = []
        used_pdb_groups = set()
        for candidate in candidates:
            pdb_group = candidate["_pdb_group"]
            if pdb_group in used_pdb_groups:
                continue
            selected.append(candidate)
            used_pdb_groups.add(pdb_group)
            if len(selected) >= args.top_n:
                break

    return candidates, selected


def write_alignments_tsv(out_path: Path, selected: List[dict]) -> None:
    out_rows = [
        {k: v for k, v in row.items() if not k.startswith("_")}
        for row in selected
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out_rows, columns=[
        "template_pdbfile",
        "target_to_template_alignstring",
        "identities",
        "target_len",
        "template_len",
    ]).to_csv(out_path, sep="\t", index=False)


def write_debug_scores_tsv(out_path: Path, candidates: List[dict], selected: List[dict]) -> None:
    debug_rows = []
    for rank, row in enumerate(candidates, start=1):
        debug_rows.append({
            "rank": rank,
            "template_id": row["_template_id"],
            "pdb_group": row["_pdb_group"],
            "template_pdbfile": row["template_pdbfile"],
            "combined_score": row["_combined_score"],
            "norm_mhc_score": row["_norm_mhc_score"],
            "norm_peptide_score": row["_norm_peptide_score"],
            "mhc_score": row["_mhc_score"],
            "mhc_identity": row["_mhc_identity"],
            "peptide_score": row["_peptide_score"],
            "peptide_mismatches": row["_peptide_mismatches"],
            "selected_top_n": row in selected,
        })
    pd.DataFrame(debug_rows).to_csv(out_path, sep="\t", index=False)


def copy_selected_templates(selected: List[dict]) -> None:
    """Copy each selected template's source PDB to its destination path,
    matching initialize.py's shutil.copy2 behavior."""
    for row in selected:
        src = row["_source_pdb_path"]
        dest = Path(row["template_pdbfile"])
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not src.exists():
            raise FileNotFoundError(f"Template PDB not found for copy: {src}")
        shutil.copy2(src, dest)


def write_target_tsv(out_path: Path, target_pdbid: str, hla_seq: str, pep_seq: str,
                      allele_name: str, start: int, alignments_path: str) -> None:
    """Same columns/row shape as initialize.py's target.tsv."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    target_chainseq = f"{hla_seq}/{pep_seq}"
    with open(out_path, "w") as tsvfile:
        tsvfile.write("mhc\tstart\tpeptide\ttargetid\ttarget_chainseq\ttemplates_alignfile\n")
        tsvfile.write(
            f"{allele_name}\t{start}\t{pep_seq}\t{target_pdbid}\t{target_chainseq}\t{alignments_path}\n"
        )


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate per-target target.tsv/alignments.tsv/templates/ (initialize.py's "
            "input format and directory layout) using the weighted MHC+peptide template "
            "selection logic from the Excel/PDB-driven version of this script."
        )
    )
    # Input locations (initialize.py convention)
    parser.add_argument("--input-seq-dir", default="input_seq",
                        help="Directory of <target_pdbid>_*.txt files (line1=hla_seq, line2=pep_seq). Default: input_seq")
    parser.add_argument("--template-pdb-dir", default="template_pdbs",
                        help="Directory of candidate template PDB structure files (chain A=MHC, chain B=9-mer peptide). Default: template_pdbs")
    parser.add_argument("--output-root", default=".",
                        help="Root directory under which <target_pdbid>/inputs/... is created. Default: current directory")

    # target.tsv fields not derivable from sequence files (same as initialize.py)
    parser.add_argument("--allele-name", default="A*02:01",
                        help="Value written to the 'mhc' column of target.tsv. Default: A*02:01 (matches initialize.py; not derived from data).")
    parser.add_argument("--start", type=int, default=0,
                        help="Value written to the 'start' column of target.tsv. Default: 0")

    # Template selection (unchanged from the Excel/PDB version)
    parser.add_argument("--top-n", type=int, default=4, help="Number of templates to write per target. Default: 4.")
    parser.add_argument("--allow-same-pdb-multiple-times", action="store_true",
                        help=("Allow multiple chain-pair variants from the same base PDB ID "
                              "among the selected templates. By default, only one template "
                              "per base PDB ID is selected, e.g. 6PTE-AC excludes 6PTE-DF."))
    parser.add_argument("--min-peptide-mismatches", type=int, default=0,
                        help="Exclude templates with peptide mismatches smaller than this value. Default: 0.")
    parser.add_argument("--disable-peptide-mismatch-filter", action="store_true",
                        help="Do not filter by peptide mismatches before ranking.")
    parser.add_argument("--max-mhc-identity", type=float, default=1.0,
                        help="Exclude templates with MHC identity greater than this value. Default: 1.0.")
    parser.add_argument("--disable-mhc-identity-filter", action="store_true",
                        help="Do not filter by MHC identity before ranking.")
    parser.add_argument("--mhc-weight", type=positive_float, default=0.3,
                        help="Weight for normalized MHC alignment score. Default: 0.3.")
    parser.add_argument("--peptide-weight", type=positive_float, default=0.7,
                        help="Weight for normalized peptide BLOSUM score. Default: 0.7.")
    parser.add_argument("--allow-fewer-than-top-n", action="store_true",
                        help="Write fewer than --top-n templates if filters remove too many candidates.")
    parser.add_argument("--write-debug-scores", action="store_true",
                        help="Also write <target_pdbid>/inputs/debug_scores.tsv with raw and normalized scores.")

    args = parser.parse_args()

    if args.top_n <= 0:
        raise SystemExit("--top-n must be positive")
    if args.mhc_weight == 0 and args.peptide_weight == 0:
        raise SystemExit("At least one of --mhc-weight or --peptide-weight must be greater than zero")

    total_weight = args.mhc_weight + args.peptide_weight
    mhc_weight = args.mhc_weight / total_weight
    peptide_weight = args.peptide_weight / total_weight

    output_root = Path(args.output_root)
    input_seq_dir = Path(args.input_seq_dir)
    template_pdb_dir = Path(args.template_pdb_dir)

    # Load candidate templates once, reading sequences directly from the PDB
    # structures (same approach as the Excel/PDB version of this script) --
    # no separate template sequence file required.
    template_paths = sorted(template_pdb_dir.glob("*.pdb"))
    if not template_paths:
        raise SystemExit(f"No .pdb files found in {template_pdb_dir}")

    template_records = []
    for tpath in template_paths:
        seqs = read_pdb_chain_sequences(tpath)
        if "A" not in seqs or "B" not in seqs:
            print(f"WARNING: skip template missing chain A or B: {tpath}")
            continue
        if len(seqs["B"]) != 9:
            print(f"WARNING: skip template with non-9mer peptide chain B: {tpath} length={len(seqs['B'])}")
            continue

        template_id = pdb_id_from_template_path(tpath)
        template_records.append({
            "id": template_id,
            "pdb_group": template_pdb_group(template_id),
            "mhc": seqs["A"],
            "pep": seqs["B"],
            "full": seqs["A"] + seqs["B"],
            "source_pdb_path": tpath,
        })

    if not template_records:
        raise SystemExit(f"No usable template PDBs found in {template_pdb_dir}")

    target_seq_files = sorted(glob.glob(str(input_seq_dir / "*.txt")))
    if not target_seq_files:
        raise SystemExit(f"No target sequence files found in {input_seq_dir}")

    for seq_file in target_seq_files:
        seq_file = Path(seq_file)
        target_pdbid = pdbid_from_seq_filename(seq_file)
        hla_seq, pep_seq = get_seq(seq_file)
        if len(pep_seq) != 9:
            print(f"WARNING: skip target with non-9mer peptide: {seq_file} length={len(pep_seq)}")
            continue

        target_dir = output_root / target_pdbid / "inputs"
        (target_dir / "templates").mkdir(parents=True, exist_ok=True)

        candidates, selected = select_templates_for_target(
            target_pdbid, hla_seq, pep_seq, template_records, args, mhc_weight, peptide_weight, target_dir
        )

        if not selected:
            if args.allow_fewer_than_top_n:
                write_alignments_tsv(target_dir / "alignments.tsv", [])
                print(f"WARNING: wrote {target_dir / 'alignments.tsv'} with 0 templates")
            else:
                raise SystemExit(
                    f"{target_pdbid}: no templates remain after filtering; "
                    f"relax filters or use --allow-fewer-than-top-n"
                )
        else:
            if len(selected) < args.top_n and not args.allow_fewer_than_top_n:
                raise SystemExit(
                    f"{target_pdbid}: only {len(selected)} templates remain after filtering; "
                    f"rerun with --allow-fewer-than-top-n or relax filters."
                )
            write_alignments_tsv(target_dir / "alignments.tsv", selected)
            copy_selected_templates(selected)
            if args.write_debug_scores:
                write_debug_scores_tsv(target_dir / "debug_scores.tsv", candidates, selected)
            print(
                f"wrote {target_dir / 'alignments.tsv'} with {len(selected)} templates "
                f"using weights mhc={mhc_weight:.3f}, peptide={peptide_weight:.3f}; "
                f"unique_base_pdb={not args.allow_same_pdb_multiple_times}"
            )

        write_target_tsv(
            target_dir / "target.tsv",
            target_pdbid=target_pdbid,
            hla_seq=hla_seq,
            pep_seq=pep_seq,
            allele_name=args.allele_name,
            start=args.start,
            alignments_path=str(target_dir / "alignments.tsv"),
        )
        print(f"wrote {target_dir / 'target.tsv'}")


if __name__ == "__main__":
    main()
