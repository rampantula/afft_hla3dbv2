#!/usr/bin/env python3
"""
Generate one AlphaFold templates_alignfile TSV per target.

Combined MHC + peptide template selection
-----------------------------------------
For each target, this script:
    1. Reads target PDB ID, allele, and peptide sequence from --excel.
    2. Infers the target MHC sequence from a template PDB assigned to the same allele
       in the Excel file (no separate --pdb-dir is required).
    3. Reads candidate template PDBs from --template-pdb-dir.
    4. Excludes:
        - self-template
        - templates missing chain A or B
        - templates with non-9mer peptide chain B
        - optionally, templates with peptide mismatches < --min-peptide-mismatches
        - optionally, templates with MHC identity > --max-mhc-identity
    5. Computes:
        - MHC Needleman-Wunsch BLOSUM62 alignment score
        - MHC sequence identity over aligned residue pairs
        - peptide positional BLOSUM62 score
        - peptide mismatches
    6. Normalizes MHC and peptide scores among candidates for each target.
    7. Ranks templates by:

        combined_score = mhc_weight * normalized_mhc_score
                       + peptide_weight * normalized_peptide_score

    8. Writes the top N templates to:
        <out-alignments-dir>/<PDB ID>__alignments.tsv

Output columns required by predict_utils.create_batch_for_training():
    template_pdbfile
    target_to_template_alignstring
    identities
    target_len
    template_len

Optional debug output can be written with --write-debug-scores.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

try:
    from Bio import pairwise2
    from Bio.Align import substitution_matrices
    BLOSUM62 = substitution_matrices.load("BLOSUM62")
except Exception as exc:
    raise SystemExit("Biopython is required: pip install biopython") from exc


AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "MSE": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y",
    "VAL": "V",
}


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


def template_pdb_group(template_id: str) -> str:
    """Return the base PDB group used to prevent duplicate template complexes.

    Examples:
        6PTE-AC -> 6PTE
        6PTE-DF -> 6PTE
        7U21    -> 7U21

    Thus, after one template from a PDB entry is selected, another chain-pair
    variant from the same base PDB entry cannot also be selected.
    """
    return template_id.split("-", 1)[0].upper()


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
    """Return best global alignment object for MHC sequences."""
    return pairwise2.align.globalds(
        target_mhc,
        template_mhc,
        BLOSUM62,
        -11,
        -1,
        one_alignment_only=True,
    )[0]


def mhc_alignment_mapping_identity_score(
    target_mhc: str,
    template_mhc: str,
) -> Tuple[Dict[int, int], float, float]:
    """Return target-to-template MHC mapping, MHC identity, and MHC alignment score."""
    aln = needleman_wunsch_mhc(target_mhc, template_mhc)
    aligned_target, aligned_template, score = aln.seqA, aln.seqB, float(aln.score)

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


def collect_templates(template_pdb_dir: Path) -> List[Path]:
    return sorted(template_pdb_dir.glob("*.pdb"))


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--excel", required=True, help="Input Excel file containing a 'PDB ID' column.")
    parser.add_argument("--template-pdb-dir", required=True, help="Directory containing candidate template PDB files.")
    parser.add_argument("--out-alignments-dir", required=True, help="Output directory for *__alignments.tsv files.")
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
    parser.add_argument("--pdb-id-column", default="PDB ID", help="Column name containing PDB IDs. Default: 'PDB ID'.")
    parser.add_argument("--allele-column", default="Allele", help="Excel column containing the target HLA allele. Default: 'Allele'.")
    parser.add_argument("--peptide-column", default="Peptide Sequence", help="Excel column containing the target peptide sequence. Default: 'Peptide Sequence'.")
    parser.add_argument("--write-debug-scores", action="store_true",
                        help="Also write <PDB ID>__debug_scores.tsv with raw and normalized scores.")
    args = parser.parse_args()

    if args.top_n <= 0:
        raise SystemExit("--top-n must be positive")
    if args.mhc_weight == 0 and args.peptide_weight == 0:
        raise SystemExit("At least one of --mhc-weight or --peptide-weight must be greater than zero")

    # Normalize weights so users can pass 6/4, 0.6/0.4, etc.
    total_weight = args.mhc_weight + args.peptide_weight
    mhc_weight = args.mhc_weight / total_weight
    peptide_weight = args.peptide_weight / total_weight

    excel = Path(args.excel)
    template_pdb_dir = Path(args.template_pdb_dir)
    out_dir = Path(args.out_alignments_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(excel, dtype=str)

    required_columns = [args.pdb_id_column, args.allele_column, args.peptide_column]
    missing_columns = [c for c in required_columns if c not in df.columns]
    if missing_columns:
        raise SystemExit(
            f"Missing required Excel column(s) {missing_columns}. "
            f"Available columns: {list(df.columns)}"
        )

    # Normalize target/template metadata from the spreadsheet.
    excel_meta = {}
    for _, row in df.iterrows():
        if pd.isna(row[args.pdb_id_column]):
            continue
        pid = str(row[args.pdb_id_column]).strip().upper()
        if not pid:
            continue
        excel_meta[pid] = {
            "allele": "" if pd.isna(row[args.allele_column]) else str(row[args.allele_column]).strip(),
            "peptide": "" if pd.isna(row[args.peptide_column]) else str(row[args.peptide_column]).strip().upper(),
        }

    template_paths = collect_templates(template_pdb_dir)
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
        pdb_group = template_pdb_group(template_id)
        meta = excel_meta.get(pdb_group, {})
        template_records.append({
            "path": tpath,
            "id": template_id,
            "pdb_group": pdb_group,
            "mhc": seqs["A"],
            "pep": seqs["B"],
            "full": seqs["A"] + seqs["B"],
            "allele": meta.get("allele", ""),
        })

    if not template_records:
        raise SystemExit("No usable template records remain after chain/peptide checks")

    for _, target_row in df.iterrows():
        if pd.isna(target_row[args.pdb_id_column]):
            continue

        pdb_id = str(target_row[args.pdb_id_column]).strip()
        if not pdb_id:
            continue

        target_group = template_pdb_group(pdb_id)
        target_allele = (
            "" if pd.isna(target_row[args.allele_column])
            else str(target_row[args.allele_column]).strip()
        )
        target_pep = (
            "" if pd.isna(target_row[args.peptide_column])
            else str(target_row[args.peptide_column]).strip().upper()
        )

        if not target_allele:
            print(f"WARNING: missing allele in Excel, skip {pdb_id}")
            continue
        if len(target_pep) != 9:
            print(
                f"WARNING: target peptide from Excel is not 9mer, "
                f"skip {pdb_id}: peptide={target_pep!r} length={len(target_pep)}"
            )
            continue

        # Infer the target MHC sequence from any available template PDB assigned
        # to the same allele in this Excel file. The target's own PDB, if present
        # in --template-pdb-dir, may be used only as a sequence reference here;
        # it is still excluded from candidate-template selection below.
        same_allele_records = [
            rec for rec in template_records
            if rec["allele"] == target_allele
        ]

        if not same_allele_records:
            print(
                f"WARNING: cannot infer target MHC sequence for {pdb_id} "
                f"({target_allele}): no template PDB in --template-pdb-dir "
                f"is assigned to this allele in the Excel file; skip"
            )
            continue

        # Choose the most commonly observed MHC sequence for the allele.
        # Ties are resolved deterministically by longer sequence, then sequence.
        sequence_counts = {}
        for rec in same_allele_records:
            sequence_counts[rec["mhc"]] = sequence_counts.get(rec["mhc"], 0) + 1
        target_mhc = sorted(
            sequence_counts,
            key=lambda seq: (sequence_counts[seq], len(seq), seq),
            reverse=True,
        )[0]

        target_full = target_mhc + target_pep

        #add
        candidates = []

        #for rec in template_records:
        #    if rec["id"] == pdb_id:
        #        continue
        for rec in template_records:
            if rec["pdb_group"] == target_group:
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

            candidates.append({
                "template_pdbfile": str(rec["path"]),
                "target_to_template_alignstring": alignstring,
                "identities": identity_count(target_full, rec["full"], full_mapping),
                "target_len": len(target_full),
                "template_len": len(rec["full"]),
                "_template_id": rec["id"],
                "_pdb_group": rec["pdb_group"],
                "_mhc_score": mhc_score,
                "_mhc_identity": mhc_identity,
                "_peptide_score": peptide_score,
                "_peptide_mismatches": mismatches,
            })

        if not candidates:
            if args.allow_fewer_than_top_n:
                out_path = out_dir / f"{pdb_id}__alignments.tsv"
                pd.DataFrame(columns=[
                    "template_pdbfile",
                    "target_to_template_alignstring",
                    "identities",
                    "target_len",
                    "template_len",
                ]).to_csv(out_path, sep="\t", index=False)
                print(f"WARNING: wrote {out_path} with 0 templates")
                continue
            raise SystemExit(f"{pdb_id}: no templates remain after filtering; relax filters or use --allow-fewer-than-top-n")

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
        if len(selected) < args.top_n and not args.allow_fewer_than_top_n:
            raise SystemExit(
                f"{pdb_id}: only {len(selected)} templates remain after filtering; "
                f"rerun with --allow-fewer-than-top-n or relax filters."
            )

        out_rows = [
            {k: v for k, v in row.items() if not k.startswith("_")}
            for row in selected
        ]
        out_path = out_dir / f"{pdb_id}__alignments.tsv"
        pd.DataFrame(out_rows, columns=[
            "template_pdbfile",
            "target_to_template_alignstring",
            "identities",
            "target_len",
            "template_len",
        ]).to_csv(out_path, sep="\t", index=False)

        if args.write_debug_scores:
            debug_path = out_dir / f"{pdb_id}__debug_scores.tsv"
            debug_rows = []
            for rank, row in enumerate(candidates, start=1):
                debug_rows.append({
                    "rank": rank,
                    "target_allele": target_allele,
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
            pd.DataFrame(debug_rows).to_csv(debug_path, sep="\t", index=False)

        print(
            f"wrote {out_path} with {len(out_rows)} templates "
            f"using weights mhc={mhc_weight:.3f}, peptide={peptide_weight:.3f}; "
            f"unique_base_pdb={not args.allow_same_pdb_multiple_times}"
        )


if __name__ == "__main__":
    main()
