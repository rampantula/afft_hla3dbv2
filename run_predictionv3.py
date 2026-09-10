#!/usr/bin/env python3
"""
Deterministic prediction with fine-tuned AlphaFold parameters.
"""

import argparse
import inspect
import itertools
import os
import pickle
import random
import sys
from pathlib import Path

import jax
import numpy as np
import pandas as pd

from alphafold.model import config
from alphafold.model import model

import predict_utils
from alphafold.common import residue_constants, protein
from alphafold.common.protein import Protein
from alphafold.model.all_atom import atom37_to_torsion_angles, atom37_to_frames
import jax.numpy as jnp
import train_utils


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run deterministic AlphaFold inference with fine-tuned weights."
    )
    parser.add_argument("--targets", required=True,
                        help="TSV containing target_chainseq and templates_alignfile.")
    parser.add_argument("--params_file", required=True,
                        help="Fine-tuned parameter pickle saved by the training script.")
    parser.add_argument("--outfile_prefix", required=True)
    parser.add_argument("--output_dir", default=".",
                        help="Directory for PDB, metric, and final TSV outputs.")
    parser.add_argument("--default_anchor_class", type=int, default=7,
                        help="Fallback peptide anchor separation used for D-score.")
    parser.add_argument("--anchor_class_file", default=None,
                        help="Optional CSV keyed by pdbid with anchor_class, anchor1, anchor2.")
    parser.add_argument("--dscore_similarity_cutoff", type=float, default=1.5)
    parser.add_argument("--model_name", default="model_2_ptm")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exact_validation", action="store_true",
                        help="Reproduce the fine-tuning validation path exactly: use create_batch_for_training, validation row index RNG, crop/trimming masks, and direct-array D-score.")
    parser.add_argument("--validation_index_col", default=None,
                        help="Optional column containing the original 0-based validation-loader index. Otherwise current TSV row order is used.")

    # These defaults match the fine-tuning script supplied in this conversation.
    parser.add_argument("--crop_size", type=int, default=190,
                        help="Defaults to the longest target. Set this to the training crop_size "
                             "when reproducing validation.")
    parser.add_argument("--msa_clusters", type=int, default=5)
    parser.add_argument("--extra_msa", type=int, default=1)
    parser.add_argument("--num_evo_blocks", type=int, default=48)
    parser.add_argument("--num_recycle", type=int, default=None,
                        help="Override AlphaFold's configured number of recycles.")
    parser.add_argument("--struc_viol_weight", type=float, default=1.0)

    parser.add_argument(
        "--resample_msa",
        action="store_true",
        help="Enable MSA resampling during recycling. The fine-tuning validation "
             "configuration enabled this; a fixed seed is still used."
    )
    parser.add_argument("--ignore_identities", action="store_true")
    parser.add_argument("--no_pdbs", action="store_true")
    parser.add_argument("--terse", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def seed_everything(seed):
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)


def load_finetuned_runner(args, crop_size):
    model_config = config.model_config(args.model_name)

    model_config.data.common.resample_msa_in_recycling = args.resample_msa
    model_config.model.resample_msa_in_recycling = args.resample_msa
    model_config.data.common.max_extra_msa = args.extra_msa
    model_config.data.eval.max_msa_clusters = args.msa_clusters
    model_config.data.eval.crop_size = crop_size
    model_config.model.embeddings_and_evoformer.evoformer_num_block = (
        args.num_evo_blocks
    )
    model_config.model.heads.structure_module.structural_violation_loss_weight = (
        args.struc_viol_weight
    )

    if args.num_recycle is not None:
        model_config.model.num_recycle = args.num_recycle
        # Some AlphaFold versions also carry this field in data.common.
        if hasattr(model_config.data.common, "num_recycle"):
            model_config.data.common.num_recycle = args.num_recycle

    with open(args.params_file, "rb") as handle:
        params = pickle.load(handle)

    runner = model.RunModel(model_config, params)
    return {args.model_name: runner}


def build_template_features(target_row, query_sequence, ignore_identities):
    alignfile = Path(target_row.templates_alignfile)
    if not alignfile.exists():
        raise FileNotFoundError(f"Template alignment file not found: {alignfile}")

    alignment_df = pd.read_table(alignfile)
    template_features_list = []

    for template_number, row in alignment_df.iterrows():
        if int(row.target_len) != len(query_sequence):
            raise ValueError(
                f"target_len mismatch in {alignfile}: "
                f"{row.target_len} != {len(query_sequence)}"
            )

        target_to_template_alignment = {
            int(pair.split(":")[0]): int(pair.split(":")[1])
            for pair in str(row.target_to_template_alignstring).split(";")
            if pair
        }

        expected_identities = None if ignore_identities else row.identities
        template_features = predict_utils.create_single_template_features(
            query_sequence,
            row.template_pdbfile,
            target_to_template_alignment,
            f"T{template_number:03d}",
            allow_chainbreaks=True,
            allow_skipped_lines=True,
            expected_identities=expected_identities,
            expected_template_len=row.template_len,
        )
        template_features_list.append(template_features)

    return predict_utils.compile_template_features(template_features_list)


def run_prediction_compatibly(**kwargs):
    """Pass a fixed seed when the local predict_utils implementation supports it."""
    signature = inspect.signature(predict_utils.run_alphafold_prediction)

    if "random_seed" in signature.parameters:
        kwargs["random_seed"] = kwargs.pop("_seed")
    elif "seed" in signature.parameters:
        kwargs["seed"] = kwargs.pop("_seed")
    else:
        # The helper may use NumPy's global RNG. seed_everything() already fixed it.
        kwargs.pop("_seed")

    return predict_utils.run_alphafold_prediction(**kwargs)



ATOM_N = residue_constants.atom_order["N"]
ATOM_CA = residue_constants.atom_order["CA"]
ATOM_C = residue_constants.atom_order["C"]


def _read_pdb_atom37(pdbfile):
    """Read one- or multi-chain PDB coordinates into ordered atom37 arrays.

    Residues are retained in their file order, so a split prediction containing
    receptor chain A followed by peptide chain B is returned as one concatenated
    residue array. This avoids alphafold.common.protein.from_pdb_string(), which
    requires a chain_id when the PDB contains multiple chains.
    """
    residue_keys = []
    key_to_index = {}
    atom_records = []

    with open(pdbfile) as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue

            # Ignore non-primary alternate conformations.
            altloc = line[16]
            if altloc not in (" ", "A"):
                continue

            atom_name = line[12:16].strip()
            if atom_name not in residue_constants.atom_order:
                continue

            chain_id = line[21]
            residue_number = int(line[22:26])
            insertion_code = line[26]
            residue_key = (chain_id, residue_number, insertion_code)

            if residue_key not in key_to_index:
                key_to_index[residue_key] = len(residue_keys)
                residue_keys.append(residue_key)

            try:
                xyz = (
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                )
            except ValueError as exc:
                raise ValueError(
                    f"Invalid coordinate record in {pdbfile}: {line.rstrip()}"
                ) from exc

            atom_records.append(
                (
                    key_to_index[residue_key],
                    residue_constants.atom_order[atom_name],
                    xyz,
                )
            )

    if not residue_keys:
        raise ValueError(f"No ATOM/HETATM residues found in PDB: {pdbfile}")

    atom_positions = np.zeros(
        (len(residue_keys), residue_constants.atom_type_num, 3),
        dtype=np.float32,
    )
    atom_mask = np.zeros(
        (len(residue_keys), residue_constants.atom_type_num),
        dtype=np.float32,
    )

    for residue_index, atom_index, xyz in atom_records:
        atom_positions[residue_index, atom_index] = xyz
        atom_mask[residue_index, atom_index] = 1.0

    return atom_positions, atom_mask


def _dihedral_degrees(p0, p1, p2, p3):
    b0 = -(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2
    norm = np.linalg.norm(b1)
    if norm < 1e-6:
        return np.nan
    b1 = b1 / norm
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    if np.linalg.norm(v) < 1e-6 or np.linalg.norm(w) < 1e-6:
        return np.nan
    return float(np.degrees(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w))))


def _cyclic_angle_distance(a, b):
    return 2.0 * (1.0 - np.cos(np.radians(a) - np.radians(b)))


def _parse_int_mapping(text):
    return {
        int(item.split(":")[0]): int(item.split(":")[1])
        for item in str(text).split(";") if item
    }


def _peptide_positions(row):
    value = row.get("peptide_positions", np.nan)
    if not pd.isna(value):
        return sorted({int(x) for x in str(value).split(";") if x})
    lengths = [len(x) for x in str(row.target_chainseq).split("/")]
    start = sum(lengths[:-1])
    return list(range(start, start + lengths[-1]))


def _load_anchor_table(path):
    if not path:
        return {}
    df = pd.read_csv(path)
    required = {"pdbid", "anchor_class", "anchor1", "anchor2"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"anchor_class_file is missing columns: {sorted(missing)}")
    out = {}
    for _, r in df.iterrows():
        item = (int(r.anchor_class), int(r.anchor1), int(r.anchor2))
        key = str(r.pdbid).strip()
        out[key] = item
        out[key.split("-")[0]] = item
    return out


def _anchor_values(row, peptide_positions, anchor_table, default_anchor_class):
    peptide_len = len(peptide_positions)
    peptide_start = min(peptide_positions)
    info = None
    for col in ("pdbid", "targetid", "native_pdbid"):
        value = row.get(col, np.nan)
        if not pd.isna(value):
            key = str(value).strip()
            info = anchor_table.get(key, anchor_table.get(key.split("-")[0]))
            if info is not None:
                break
    if info is None:
        anchor_class = int(row.get("anchor_class", default_anchor_class))
        if "anchor1" in row and "anchor2" in row and not pd.isna(row.anchor1) and not pd.isna(row.anchor2):
            raw1, raw2 = int(row.anchor1), int(row.anchor2)
        else:
            raw2 = peptide_len
            raw1 = max(1, raw2 - anchor_class)
    else:
        anchor_class, raw1, raw2 = info

    def local(raw):
        return raw if 1 <= raw <= peptide_len else raw - peptide_start

    anchor1, anchor2 = local(raw1), local(raw2)
    if not (1 <= anchor1 < anchor2 <= peptide_len):
        raise ValueError(f"Invalid peptide anchors: {anchor1}, {anchor2}; length={peptide_len}")
    return float(anchor_class), anchor1, anchor2


def _find_prediction_pdb(row_prefix, model_name):
    prefix = Path(row_prefix)
    candidates = sorted(prefix.parent.glob(prefix.name + "*.pdb"))
    model_hits = [p for p in candidates if model_name in p.name]
    if model_hits:
        return model_hits[-1]
    return candidates[-1] if candidates else None



ATOM_RECORDS = ("ATOM  ", "HETATM")


def _pdb_residue_key(line):
    """Return the residue identity from an ATOM/HETATM PDB line."""
    return (line[21], int(line[22:26]), line[26])


def _replace_chain_and_resseq(line, chain_id, residue_number):
    """Replace PDB chain ID, residue number, and insertion code."""
    return line[:21] + chain_id + f"{residue_number:4d}" + " " + line[27:]


def split_prediction_pdb(input_pdb, output_pdb, target_chainseq):
    """Split an AlphaFold prediction into receptor chain A and peptide chain B.

    The final slash-separated sequence in target_chainseq is treated as the
    peptide. All preceding residues are assigned to receptor chain A. Residue
    order in the input PDB must match target_chainseq.
    """
    sequences = str(target_chainseq).split("/")
    if len(sequences) < 2:
        raise ValueError(
            "target_chainseq must contain at least two slash-separated chains "
            "so the final chain can be assigned as peptide chain B."
        )

    peptide_len = len(sequences[-1])
    receptor_len = sum(len(sequence) for sequence in sequences[:-1])
    expected_len = receptor_len + peptide_len

    residue_keys = []
    seen = set()
    with open(input_pdb) as handle:
        for line in handle:
            if not line.startswith(ATOM_RECORDS):
                continue
            key = _pdb_residue_key(line)
            if key not in seen:
                seen.add(key)
                residue_keys.append(key)

    if len(residue_keys) != expected_len:
        raise ValueError(
            f"{input_pdb}: found {len(residue_keys)} residues, but "
            f"target_chainseq expects {expected_len} "
            f"({receptor_len} receptor + {peptide_len} peptide)."
        )

    key_to_new = {}
    for index, key in enumerate(residue_keys):
        if index < receptor_len:
            key_to_new[key] = ("A", index + 1)
        else:
            key_to_new[key] = ("B", index - receptor_len + 1)

    output_pdb = Path(output_pdb)
    output_pdb.parent.mkdir(parents=True, exist_ok=True)

    last_chain = None
    with open(input_pdb) as source, open(output_pdb, "w") as destination:
        destination.write("REMARK Split prediction: receptor chain A, peptide chain B\n")
        destination.write(f"REMARK Receptor length: {receptor_len}\n")
        destination.write(f"REMARK Peptide length: {peptide_len}\n")

        for line in source:
            if not line.startswith(ATOM_RECORDS):
                continue

            key = _pdb_residue_key(line)
            chain_id, residue_number = key_to_new[key]

            if last_chain is not None and chain_id != last_chain:
                destination.write("TER\n")
            last_chain = chain_id

            destination.write(
                _replace_chain_and_resseq(line, chain_id, residue_number)
            )

        destination.write("TER\nEND\n")

    return output_pdb

def compute_output_dscore(predicted_pdb, row, anchor_table, default_anchor_class):
    """Compute the central, scaled HLA3DB-style D-score versus native structure."""
    native_pdb = row.get("native_pdbfile", np.nan)
    native_align = row.get("native_alignstring", np.nan)
    if pd.isna(native_pdb) or pd.isna(native_align):
        return np.nan

    pred_pos, pred_mask = _read_pdb_atom37(predicted_pdb)
    native_pos, native_mask = _read_pdb_atom37(str(native_pdb))
    target_to_native = _parse_int_mapping(native_align)
    pep_positions = _peptide_positions(row)
    anchor_class, anchor1, anchor2 = _anchor_values(
        row, pep_positions, anchor_table, default_anchor_class)
    central_local = set(range(anchor1 + 2, anchor2 - 1))

    angle_sum = 0.0
    angle_count = 0
    for local_pos, target_i in enumerate(pep_positions, start=1):
        if local_pos not in central_local or target_i not in target_to_native:
            continue
        native_i = target_to_native[target_i]

        # phi_i
        if (target_i - 1 in target_to_native and target_i > 0 and
                native_i > 0 and target_to_native[target_i - 1] == native_i - 1):
            req_p = [(target_i - 1, ATOM_C), (target_i, ATOM_N), (target_i, ATOM_CA), (target_i, ATOM_C)]
            req_n = [(native_i - 1, ATOM_C), (native_i, ATOM_N), (native_i, ATOM_CA), (native_i, ATOM_C)]
            if all(r < len(pred_mask) and pred_mask[r, a] > 0.5 for r, a in req_p) and all(r < len(native_mask) and native_mask[r, a] > 0.5 for r, a in req_n):
                pa = _dihedral_degrees(pred_pos[target_i-1, ATOM_C], pred_pos[target_i, ATOM_N], pred_pos[target_i, ATOM_CA], pred_pos[target_i, ATOM_C])
                na = _dihedral_degrees(native_pos[native_i-1, ATOM_C], native_pos[native_i, ATOM_N], native_pos[native_i, ATOM_CA], native_pos[native_i, ATOM_C])
                if np.isfinite(pa) and np.isfinite(na):
                    angle_sum += _cyclic_angle_distance(pa, na); angle_count += 1

        # psi_i
        if (target_i + 1 in target_to_native and target_i + 1 < len(pred_pos) and
                native_i + 1 < len(native_pos) and target_to_native[target_i + 1] == native_i + 1):
            req_p = [(target_i, ATOM_N), (target_i, ATOM_CA), (target_i, ATOM_C), (target_i + 1, ATOM_N)]
            req_n = [(native_i, ATOM_N), (native_i, ATOM_CA), (native_i, ATOM_C), (native_i + 1, ATOM_N)]
            if all(pred_mask[r, a] > 0.5 for r, a in req_p) and all(native_mask[r, a] > 0.5 for r, a in req_n):
                pa = _dihedral_degrees(pred_pos[target_i, ATOM_N], pred_pos[target_i, ATOM_CA], pred_pos[target_i, ATOM_C], pred_pos[target_i+1, ATOM_N])
                na = _dihedral_degrees(native_pos[native_i, ATOM_N], native_pos[native_i, ATOM_CA], native_pos[native_i, ATOM_C], native_pos[native_i+1, ATOM_N])
                if np.isfinite(pa) and np.isfinite(na):
                    angle_sum += _cyclic_angle_distance(pa, na); angle_count += 1

    num_angles = anchor_class * 2.0 - 6.0
    if angle_count == 0 or num_angles <= 0:
        return np.nan
    return float(angle_sum * 8.0 / num_angles)



# ---------------- Exact fine-tuning validation compatibility ----------------

def _get_peptide_positions_exact(row):
    value = row.get("peptide_positions", np.nan)
    if not pd.isna(value):
        return sorted({int(x) for x in str(value).split(";") if x})
    lengths = [len(x) for x in str(row.target_chainseq).split("/")]
    start = sum(lengths[:-1])
    return list(range(start, start + lengths[-1]))


def _anchor_values_exact(row, peptide_positions, anchor_table, default_anchor_class):
    peptide_len = len(peptide_positions)
    peptide_start = min(peptide_positions)
    info = None
    for col in ("pdbid", "targetid", "native_pdbid"):
        value = row.get(col, np.nan)
        if not pd.isna(value):
            key = str(value).strip()
            info = anchor_table.get(key, anchor_table.get(key.split("-")[0]))
            if info is not None:
                break
    if info is None:
        anchor_class = int(row.get("anchor_class", default_anchor_class))
        if all(c in row.index for c in ("anchor1", "anchor2")) and not pd.isna(row.anchor1) and not pd.isna(row.anchor2):
            raw1, raw2 = int(row.anchor1), int(row.anchor2)
        else:
            raw2 = peptide_len
            raw1 = max(1, raw2 - anchor_class)
    else:
        anchor_class, raw1, raw2 = info

    def local(raw):
        return raw if 1 <= raw <= peptide_len else raw - peptide_start

    a1, a2 = local(raw1), local(raw2)
    if not (1 <= a1 < a2 <= peptide_len):
        raise ValueError(f"Invalid anchors: raw=({raw1},{raw2}), local=({a1},{a2}), peptide_len={peptide_len}")
    return float(anchor_class), a1, a2


def _exact_masks(row, target_trim_positions, crop_size, anchor_table, default_anchor_class):
    peptide_positions = _get_peptide_positions_exact(row)
    peptide_set = set(peptide_positions)
    full_to_local = {p: i + 1 for i, p in enumerate(peptide_positions)}
    anchor_class, a1, a2 = _anchor_values_exact(row, peptide_positions, anchor_table, default_anchor_class)
    center_start, center_end = a1 + 2, a2 - 2
    peptide_mask = np.zeros((crop_size,), np.float32)
    central_mask = np.zeros((crop_size,), np.float32)
    for trim_pos, full_pos in enumerate(target_trim_positions):
        if trim_pos >= crop_size:
            break
        if full_pos in peptide_set:
            peptide_mask[trim_pos] = 1.0
            local = full_to_local[full_pos]
            if center_start <= local <= center_end:
                central_mask[trim_pos] = 1.0
    return peptide_mask, central_mask, np.array(anchor_class, np.float32)


def create_exact_validation_sample(row, crop_size, runner, anchor_table, default_anchor_class, debug=False, verbose=False):
    nres = len(str(row.target_chainseq).replace("/", ""))
    value = row.get("target_trim_positions", np.nan)
    target_trim_positions = (
        [int(x) for x in str(value).split(";") if x]
        if not pd.isna(value) else list(range(nres))
    )
    native_align = _parse_int_mapping(row.native_alignstring)
    native_identities = row.get("native_identities", None)
    native_identities = None if pd.isna(native_identities) else native_identities
    native_len = row.get("native_len", None)
    native_len = None if pd.isna(native_len) else native_len

    sample = predict_utils.create_batch_for_training(
        row.target_chainseq, target_trim_positions, row.templates_alignfile,
        row.native_pdbfile, native_align, crop_size, runner,
        native_identities=native_identities, native_len=native_len,
        debug=debug, verbose=verbose, random_seed=0)
    sample["peptide_mask"], sample["central_dscore_mask"], sample["anchor_class"] = _exact_masks(
        row, target_trim_positions, crop_size, anchor_table, default_anchor_class)
    return sample


def collate_exact_single(sample):
    out = {}
    for name in train_utils.list_a + train_utils.list_a_templates:
        out[name] = np.stack([sample[name][0]], axis=0)
    for name in train_utils.list_b + train_utils.list_b_templates:
        out[name] = np.stack([sample[name]], axis=0)
    out["aatype_"] = np.stack([sample["aatype"][0]], axis=0)
    out["peptide_mask"] = np.stack([sample["peptide_mask"]], axis=0)
    out["central_dscore_mask"] = np.stack([sample["central_dscore_mask"]], axis=0)
    out["anchor_class"] = np.stack([sample["anchor_class"]], axis=0)
    return out


def prep_exact_batch(batch):
    torsion = atom37_to_torsion_angles(
        jnp.array(batch["aatype_"]), jnp.array(batch["all_atom_positions"]), jnp.array(batch["all_atom_mask"]))
    batch["chi_mask"] = torsion["torsion_angles_mask"][:, :, 3:]
    batch["chi_angles"] = jnp.arctan2(
        torsion["torsion_angles_sin_cos"][:, :, 3:, 0],
        torsion["torsion_angles_sin_cos"][:, :, 3:, 1])
    batch.update(atom37_to_frames(
        jnp.array(batch["aatype_"]), jnp.array(batch["all_atom_positions"]), jnp.array(batch["all_atom_mask"])))
    for key, value in list(batch.items()):
        if key in train_utils.list_a + train_utils.list_a_templates:
            batch[key] = value[:, None]
        if key in train_utils.list_c:
            batch[key] = value[:, None]
    for key in train_utils.pdb_key_list_int:
        batch[key] = jnp.array(batch[key], jnp.int32)
    batch["peptide_mask"] = jnp.array(batch["peptide_mask"], jnp.float32)
    batch["central_dscore_mask"] = jnp.array(batch["central_dscore_mask"], jnp.float32)
    batch["anchor_class"] = jnp.array(batch["anchor_class"], jnp.float32)
    # pmap in training removes this leading axis. Do the same explicitly.
    return jax.tree_util.tree_map(lambda x: x[0], batch)


def _single_structure_angles_exact(atom_positions, atom_mask, peptide_mask, central_mask, residue_index):
    atom_positions, atom_mask = np.asarray(atom_positions), np.asarray(atom_mask)
    peptide_mask, central_mask, residue_index = map(np.asarray, (peptide_mask, central_mask, residue_index))
    while peptide_mask.ndim > 1: peptide_mask = peptide_mask[0]
    while central_mask.ndim > 1: central_mask = central_mask[0]
    while residue_index.ndim > 1: residue_index = residue_index[0]
    angles = []
    for res in range(atom_positions.shape[0]):
        if central_mask[res] < .5 or peptide_mask[res] < .5: continue
        if res > 0 and peptide_mask[res-1] >= .5 and residue_index[res]-residue_index[res-1] == 1:
            req=[(res-1,ATOM_C),(res,ATOM_N),(res,ATOM_CA),(res,ATOM_C)]
            if all(atom_mask[r,a] >= .5 for r,a in req):
                x=_dihedral_degrees(atom_positions[res-1,ATOM_C],atom_positions[res,ATOM_N],atom_positions[res,ATOM_CA],atom_positions[res,ATOM_C])
                if np.isfinite(x): angles.append(("phi",res,x))
        if res+1 < atom_positions.shape[0] and peptide_mask[res+1] >= .5 and residue_index[res+1]-residue_index[res] == 1:
            req=[(res,ATOM_N),(res,ATOM_CA),(res,ATOM_C),(res+1,ATOM_N)]
            if all(atom_mask[r,a] >= .5 for r,a in req):
                x=_dihedral_degrees(atom_positions[res,ATOM_N],atom_positions[res,ATOM_CA],atom_positions[res,ATOM_C],atom_positions[res+1,ATOM_N])
                if np.isfinite(x): angles.append(("psi",res,x))
    return angles


def compute_exact_validation_dscore(batch, predicted_dict):
    pred_pos=np.asarray(predicted_dict["structure_module"]["final_atom_positions"])
    pred_mask=np.asarray(predicted_dict["structure_module"]["final_atom_mask"])
    native_pos=np.asarray(batch["all_atom_positions"]); native_mask=np.asarray(batch["all_atom_mask"])
    while pred_pos.ndim>3: pred_pos=pred_pos[0]; pred_mask=pred_mask[0]
    while native_pos.ndim>3: native_pos=native_pos[0]; native_mask=native_mask[0]
    na=_single_structure_angles_exact(native_pos,native_mask,batch["peptide_mask"],batch["central_dscore_mask"],batch["residue_index"])
    pa=_single_structure_angles_exact(pred_pos,pred_mask,batch["peptide_mask"],batch["central_dscore_mask"],batch["residue_index"])
    nd={(k,r):a for k,r,a in na}; pd_={(k,r):a for k,r,a in pa}
    keys=set(nd).intersection(pd_)
    if not keys: return np.nan
    anchor=float(np.asarray(batch["anchor_class"]).reshape(-1)[0]); denom=anchor*2.0-6.0
    if denom<=0: return np.nan
    return float(sum(_cyclic_angle_distance(pd_[k],nd[k]) for k in keys)*8.0/denom)


def protein_from_exact_prediction(batch, predicted_dict):
    pos=np.asarray(predicted_dict["structure_module"]["final_atom_positions"])
    mask=np.asarray(predicted_dict["structure_module"]["final_atom_mask"])
    while pos.ndim>3: pos=pos[0]; mask=mask[0]
    aatype=np.asarray(batch["aatype"]); residue_index=np.asarray(batch["residue_index"])+1
    while aatype.ndim>1: aatype=aatype[0]
    while residue_index.ndim>1: residue_index=residue_index[0]
    return Protein(aatype=aatype, atom_positions=pos, atom_mask=mask,
                   residue_index=residue_index, b_factors=np.zeros_like(mask))


def main():
    args = parse_args()
    seed_everything(args.seed)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    anchor_table = _load_anchor_table(args.anchor_class_file)

    targets = pd.read_table(args.targets)
    required = {"target_chainseq", "templates_alignfile"}
    missing = required.difference(targets.columns)
    if missing:
        raise ValueError(f"Targets TSV is missing columns: {sorted(missing)}")

    target_lengths = [
        len(str(row.target_chainseq).replace("/", ""))
        for row in targets.itertuples()
    ]
    if args.exact_validation and args.crop_size is None:
        raise ValueError("--exact_validation requires the training --crop_size (190 in your fine-tuning run).")
    if args.exact_validation:
        args.resample_msa = True
    crop_size = args.crop_size or max(target_lengths)

    if max(target_lengths) > crop_size:
        raise ValueError(
            f"At least one target has {max(target_lengths)} residues, "
            f"which exceeds crop_size={crop_size}."
        )

    model_runners = load_finetuned_runner(args, crop_size)

    if args.verbose:
        print("command:", " ".join(sys.argv))
        print("device:", jax.local_devices()[0])
        print("model_name:", args.model_name)
        print("params_file:", args.params_file)
        print("seed:", args.seed)
        print("crop_size:", crop_size)
        print("resample_msa:", args.resample_msa)

    final_rows = []

    for counter, target_row in targets.iterrows():
        target_id = str(target_row.get("targetid", f"T{counter}"))
        print(f"START: {counter + 1}/{len(targets)} {target_id}", flush=True)

        if args.exact_validation:
            for required_col in ("native_pdbfile", "native_alignstring"):
                if required_col not in targets.columns:
                    raise ValueError(f"--exact_validation requires column: {required_col}")
            validation_index = (
                int(target_row[args.validation_index_col])
                if args.validation_index_col else counter
            )
            runner = model_runners[args.model_name]
            sample = create_exact_validation_sample(
                target_row, crop_size, runner, anchor_table,
                args.default_anchor_class, args.verbose, args.verbose)
            exact_batch = prep_exact_batch(collate_exact_single(sample))
            exact_key = jax.random.fold_in(jax.random.PRNGKey(0), validation_index)
            model_batch = dict(exact_batch)
            for bookkeeping_key in ("peptide_mask", "central_dscore_mask", "anchor_class"):
                model_batch.pop(bookkeeping_key, None)
            predicted_dict, _ = runner.apply(runner.params, exact_key, model_batch)
            dscore = compute_exact_validation_dscore(exact_batch, predicted_dict)

            output_row = target_row.copy()
            output_row[f"{args.model_name}_dscore"] = dscore
            output_row[f"{args.model_name}_dscore_similar"] = (
                bool(dscore <= args.dscore_similarity_cutoff) if np.isfinite(dscore) else np.nan)
            output_row["validation_index_used"] = validation_index

            if not (args.no_pdbs or args.terse):
                pdb_id = str(target_row.get("pdbid", target_id)).strip()
                safe_id = "".join(c if c.isalnum() or c in "-_." else "_" for c in pdb_id)
                raw_pdb = output_dir / f"{safe_id}_exact_validation_raw.pdb"
                with open(raw_pdb, "w") as handle:
                    handle.write(protein.to_pdb(protein_from_exact_prediction(exact_batch, predicted_dict)))
                output_row["raw_predicted_pdbfile"] = str(raw_pdb)
                try:
                    split_pdb = output_dir / f"{safe_id}_model_split.pdb"
                    split_prediction_pdb(raw_pdb, split_pdb, str(target_row.target_chainseq))
                    output_row["predicted_pdbfile"] = str(split_pdb)
                    print("made:", split_pdb)
                except Exception as exc:
                    print(f"WARNING: exact-validation PDB split failed for {target_id}: {exc}", file=sys.stderr)

            # These are useful diagnostics, but D-score above is computed directly
            # from arrays exactly as in fine-tuning validation.
            try:
                output_row[f"{args.model_name}_plddt"] = float(
                    np.mean(np.asarray(predicted_dict["predicted_lddt"]["lddt_ca"])))
            except Exception:
                pass
            final_rows.append(output_row)
            print(f"exact_validation index={validation_index} dscore={dscore}", flush=True)
            continue

        query_chainseq = str(target_row.target_chainseq)
        query_sequence = query_chainseq.replace("/", "")
        template_features = build_template_features(
            target_row, query_sequence, args.ignore_identities
        )

        # This matches the original prediction script: query-only MSA.
        msa = [query_sequence]
        deletion_matrix = [[0] * len(query_sequence)]

        row_prefix_value = target_row.get(
            "outfile_prefix", f"{args.outfile_prefix}_{target_id}"
        )
        row_prefix = str(output_dir / Path(str(row_prefix_value)).name)

        # Use a stable per-target key. Results do not depend on input row order
        # when targetid remains unchanged.
        target_seed = int(
            np.uint32(args.seed) ^
            np.uint32(sum((i + 1) * ord(c) for i, c in enumerate(target_id)))
        )
        seed_everything(target_seed)

        all_metrics = run_prediction_compatibly(
            query_sequence=query_sequence,
            msa=msa,
            deletion_matrix=deletion_matrix,
            chainbreak_sequence=query_chainseq,
            template_features=template_features,
            model_runners=model_runners,
            out_prefix=row_prefix,
            crop_size=crop_size,
            dump_pdbs=not (args.no_pdbs or args.terse),
            dump_metrics=not args.terse,
            _seed=target_seed,
        )

        output_row = target_row.copy()
        predicted_pdb = _find_prediction_pdb(row_prefix, args.model_name)
        if predicted_pdb is not None:
            output_row["raw_predicted_pdbfile"] = str(predicted_pdb)

            pdb_id = str(
                target_row.get(
                    "pdbid",
                    target_row.get("targetid", target_id),
                )
            ).strip()
            safe_pdb_id = "".join(
                char if char.isalnum() or char in {"-", "_", "."} else "_"
                for char in pdb_id
            )
            split_pdb = output_dir / f"{safe_pdb_id}_model_split.pdb"

            try:
                split_prediction_pdb(
                    predicted_pdb,
                    split_pdb,
                    query_chainseq,
                )
                print("made:", split_pdb)
                output_row["predicted_pdbfile"] = str(split_pdb)

                # D-score is calculated only after the PDB has been rewritten
                # with receptor chain A and peptide chain B.
                dscore = compute_output_dscore(
                    split_pdb,
                    target_row,
                    anchor_table,
                    args.default_anchor_class,
                )
            except Exception as exc:
                print(
                    f"WARNING: PDB splitting or D-score failed for "
                    f"{target_id}: {exc}",
                    file=sys.stderr,
                )
                dscore = np.nan

            output_row[f"{args.model_name}_dscore"] = dscore
            output_row[f"{args.model_name}_dscore_similar"] = (
                bool(dscore <= args.dscore_similarity_cutoff)
                if np.isfinite(dscore)
                else np.nan
            )
        else:
            output_row[f"{args.model_name}_dscore"] = np.nan
            output_row[f"{args.model_name}_dscore_similar"] = np.nan
            if not (args.no_pdbs or args.terse):
                print(
                    f"WARNING: no predicted PDB found for prefix {row_prefix}",
                    file=sys.stderr,
                )
        chains = query_chainseq.split("/")
        chain_stops = list(itertools.accumulate(len(chain) for chain in chains))
        chain_starts = [0] + chain_stops[:-1]
        nres = chain_stops[-1]

        for model_name, metrics in all_metrics.items():
            plddt = np.asarray(metrics["plddt"])
            pae = metrics.get("predicted_aligned_error")
            pae = None if pae is None else np.asarray(pae)

            output_row[f"{model_name}_plddt"] = float(np.mean(plddt[:nres]))
            if pae is not None:
                output_row[f"{model_name}_pae"] = float(
                    np.mean(pae[:nres, :nres])
                )

            for chain1, (start1, stop1) in enumerate(
                zip(chain_starts, chain_stops)
            ):
                output_row[f"{model_name}_plddt_{chain1}"] = float(
                    np.mean(plddt[start1:stop1])
                )
                if pae is not None:
                    for chain2, (start2, stop2) in enumerate(
                        zip(chain_starts, chain_stops)
                    ):
                        output_row[
                            f"{model_name}_pae_{chain1}_{chain2}"
                        ] = float(np.mean(pae[start1:stop1, start2:stop2]))

        final_rows.append(output_row)

    outfile = str(output_dir / f"{Path(args.outfile_prefix).name}_final.tsv")
    pd.DataFrame(final_rows).to_csv(outfile, sep="\t", index=False)
    print("made:", outfile)


if __name__ == "__main__":
    main()
