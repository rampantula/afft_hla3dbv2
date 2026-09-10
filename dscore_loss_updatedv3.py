########################################################################################
FREDHUTCH_HACKS = False

if FREDHUTCH_HACKS:
    import os
    os.environ['XLA_FLAGS']='--xla_gpu_force_compilation_parallelism=1'

import os
from os import popen
import sys
import pandas as pd
import numpy as np
import pickle
import optax
import jax
import jax.numpy as jnp
import haiku as hk
from alphafold.common import protein
from alphafold.common import confidence
from alphafold.common import residue_constants
from alphafold.common.protein import Protein
from alphafold.data import pipeline
from alphafold.data import templates
from alphafold.model import data
from alphafold.model import config
from alphafold.model import model
from alphafold.model.all_atom import atom37_to_torsion_angles, atom37_to_frames
import torch # dataloader/dataset stuff
import tensorflow.compat.v1 as tf1 # cmdline args
import warnings
warnings.filterwarnings("ignore")

## fine-tuning specific helper codes
import train_utils
import predict_utils
print('done importing') ; sys.stdout.flush()

flags = tf1.app.flags

flags.DEFINE_string('outprefix', 'testrun', help='string prefix for all outfiles')
flags.DEFINE_string('model_name', 'model_2_ptm', help='like model_1 or model_2_ptm')
flags.DEFINE_string('train_dataset', None, help='tsv file with training dataset. See '
                    'README for format info')
flags.DEFINE_string('valid_dataset', None, help='tsv file with validation dataset')
flags.DEFINE_integer('crop_size', 190, help='Max size of training example; set this '
                     'as low as possible for memory and speed')
flags.DEFINE_integer('num_epochs', 5, help='number of epochs')
flags.DEFINE_integer('batch_size', 1, help='total batch size')
flags.DEFINE_integer('apply_every', 1, help='how often to apply gradient updates')
flags.DEFINE_bool('notrain', False, help='if True, dont do any training')
flags.DEFINE_bool('debug', False, help='debug')
flags.DEFINE_bool('verbose', False, help='verbose')
flags.DEFINE_bool('test_load', False,
                  help='if True, loop through datasets to test loading')
flags.DEFINE_bool('dump_pdbs', False, help='if True, write out PDB files of '
                  'modeled structures during training')
flags.DEFINE_integer('print_steps', 25,
                     help='printed averaged results every print_steps')
flags.DEFINE_integer('save_steps', 207, help='save model + optimizer every save_steps')
flags.DEFINE_integer('valid_steps', 100000,
                     help='calc loss on whole valid set every valid_steps')
flags.DEFINE_integer('num_cpus', 0, help='number of (extra?) cpus for loading')
flags.DEFINE_float('lr_coef', 0.025, help='learning rate coefficient')
flags.DEFINE_float('fake_native_weight', 0.25, help='weight to apply to the alphafold '
                   'loss when using a predicted structure as the native')
flags.DEFINE_float('struc_viol_weight', 1.0, help='structural violation weight' 'penalizes physically impossible structures')
flags.DEFINE_float('dscore_weight', 0.1, help='weight for peptide D-score loss')
flags.DEFINE_integer('default_anchor_class', 7, help='Default peptide anchor separation used for HLA3DB central D-score when dataset lacks anchor1/anchor2 columns')
flags.DEFINE_string('anchor_class_file', 'anchor_class.csv', help='CSV file with HLA3DB anchor_class, anchor1, and anchor2 columns keyed by pdbid')
flags.DEFINE_integer('msa_clusters', 5, help='number of msa cluster sequences')
flags.DEFINE_integer('extra_msa', 1, help='number of extra msa sequences')
flags.DEFINE_integer('num_evo_blocks', 48, help='number of evoformer blocks')
flags.DEFINE_float('grad_norm_clip', 10.0, help='value to clip gradient norms, '
                   'per update')
flags.DEFINE_string('data_dir', "/home/pbradley/csdat/alphafold/data/",
                    help='location of alphafold params; passed to '
                    'data.get_model_haiku_params; should contain params/ subfolder')

flags.DEFINE_bool('freeze_everything', False, help='if True, dont fit anything')
flags.DEFINE_bool('no_ramp', False, help='if True, dont ramp')
flags.DEFINE_bool('no_valid', False, help='if True, dont compute valid stats')
flags.DEFINE_bool('no_random', False, help='if True, dont randomize')
flags.DEFINE_bool('random_recycling', False, help='if True, set num_iter_recycling '
                  'randomly during training')
flags.DEFINE_bool('calc_valid_dscore', True, help='if True, compute peptide D-score during validation')
flags.DEFINE_float('dscore_similarity_cutoff', 1.5, help='D-score cutoff for counting native/predicted as similar')
flags.DEFINE_bool('dump_valid_pdbs', False, help='if True, write predicted PDB files during validation')
flags.DEFINE_string('valid_pdb_dir', 'validation_predicted_pdbs', help='directory for validation predicted PDB files')
flags.DEFINE_string('valid_pdb_suffix', '_model', help='suffix added to validation PDB filenames, e.g. 7WZZ_model.pdb')
FLAGS = flags.FLAGS

assert FLAGS.msa_clusters >= 5 # since reduce_msa_clusters_by_max_templates is True for model_1/model_1_ptm

batch_size = FLAGS.batch_size
if batch_size>1:
    print('WARNING\n'*12,'phil has not tested batch_size>1')

assert FLAGS.apply_every >= 1

jax_key = jax.random.PRNGKey(0)
model_name = FLAGS.model_name

platform = jax.local_devices()[0].platform
hostname = popen('hostname').readlines()[0].strip()


print('cmd:', ' '.join(sys.argv))
print('local_device:', platform, hostname)
print('model_name:', model_name)
print('outprefix:', FLAGS.outprefix)
sys.stdout.flush()

model_config = config.model_config(model_name)
model_config.data.common.resample_msa_in_recycling = True
model_config.model.resample_msa_in_recycling = True
model_config.data.common.max_extra_msa = FLAGS.extra_msa
model_config.data.eval.max_msa_clusters = FLAGS.msa_clusters
model_config.data.eval.crop_size = FLAGS.crop_size
model_config.model.heads.structure_module.structural_violation_loss_weight = FLAGS.struc_viol_weight
model_config.model.embeddings_and_evoformer.evoformer_num_block = FLAGS.num_evo_blocks

### AlphaFold + D-score fine-tuning ###################################################
# In this version, the binder classifier has been removed entirely.
# The trainable parameters are only the AlphaFold parameters. The objective is the
# AlphaFold structure loss plus a differentiable peptide-backbone D-score loss,
# optionally down-weighted for predicted/fake natives.

af2_model_params = data.get_model_haiku_params(
    model_name=model_name, data_dir=FLAGS.data_dir)

model_params = af2_model_params
model_runner = model.RunModel(model_config, af2_model_params)

############################ end of AlphaFold-only setup ##############################

ATOM_N = residue_constants.atom_order['N']
ATOM_CA = residue_constants.atom_order['CA']
ATOM_C = residue_constants.atom_order['C']


def _strip_atom_ensemble_dim(x):
    return x[:, 0] if len(x.shape) == 5 else x


def _strip_mask_ensemble_dim(x):
    return x[:, 0] if len(x.shape) == 4 else x


def _strip_residue_ensemble_dim(x):
    return x[:, 0] if len(x.shape) == 3 else x


def _ensure_atom_batch_dim(x):
    return x[None, ...] if len(x.shape) == 3 else x


def _ensure_mask_batch_dim(x):
    return x[None, ...] if len(x.shape) == 2 else x


def _ensure_residue_batch_dim(x):
    return x[None, ...] if len(x.shape) == 1 else x


def _safe_normalize(v, eps=1e-8):
    return v / jnp.sqrt(jnp.maximum(jnp.sum(v * v, axis=-1, keepdims=True), eps))


def _dihedral_sin_cos(p0, p1, p2, p3, eps=1e-8):
    """Return stable sin/cos representation of a dihedral angle.

    This avoids differentiating through atan2(), whose gradient can become NaN
    when the projected vectors are near zero. The D-score only needs
    cos(pred-native), so sin/cos is sufficient and more stable for training.
    """
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2

    b1 = _safe_normalize(b1, eps)
    v = b0 - jnp.sum(b0 * b1, axis=-1, keepdims=True) * b1
    w = b2 - jnp.sum(b2 * b1, axis=-1, keepdims=True) * b1

    v = _safe_normalize(v, eps)
    w = _safe_normalize(w, eps)

    cos_angle = jnp.sum(v * w, axis=-1)
    sin_angle = jnp.sum(jnp.cross(b1, v) * w, axis=-1)

    # Numerical guard. Clipping keeps the representation bounded and avoids
    # occasional tiny roundoff excursions outside [-1, 1].
    cos_angle = jnp.clip(cos_angle, -1.0 + 1e-6, 1.0 - 1e-6)
    sin_angle = jnp.clip(sin_angle, -1.0 + 1e-6, 1.0 - 1e-6)
    return sin_angle, cos_angle


def _angle_dscore_sin_cos(pred_sin, pred_cos, native_sin, native_cos):
    """HLA3DB distance formula using sin/cos torsions.

    Equivalent to 2 * (1 - cos(pred_angle - native_angle)), but without atan2.
    """
    cos_delta = pred_cos * native_cos + pred_sin * native_sin
    cos_delta = jnp.clip(cos_delta, -1.0 + 1e-6, 1.0 - 1e-6)
    return 2.0 * (1.0 - cos_delta)


def peptide_dscore_loss(predicted_dict, processed_feature_dict, peptide_mask, central_dscore_mask, anchor_class):
    """HLA3DB-style central, scaled D-score peptide torsion loss.

    HLA3DB read_central_dihedral() compares only residues anchor1+2 through
    anchor2-2, using both phi and psi for each selected central residue. The
    resulting sum is scaled to a 14-angle equivalent, matching:
        dscore_scaled = dscore * 14 / num_angles
    in samepep_diffhla.py.
    """
    pred_pos = predicted_dict['structure_module']['final_atom_positions']
    pred_mask = predicted_dict['structure_module']['final_atom_mask']
    native_pos = _strip_atom_ensemble_dim(processed_feature_dict['all_atom_positions'])
    native_mask = _strip_mask_ensemble_dim(processed_feature_dict['all_atom_mask'])
    residue_index = _strip_residue_ensemble_dim(processed_feature_dict['residue_index'])
    peptide_mask = jnp.asarray(peptide_mask, jnp.float32)
    central_dscore_mask = jnp.asarray(central_dscore_mask, jnp.float32)
    anchor_class = jnp.asarray(anchor_class, jnp.float32)
    pred_pos = _ensure_atom_batch_dim(pred_pos)
    pred_mask = _ensure_mask_batch_dim(pred_mask)
    native_pos = _ensure_atom_batch_dim(native_pos)
    native_mask = _ensure_mask_batch_dim(native_mask)
    residue_index = _ensure_residue_batch_dim(residue_index)
    peptide_mask = _ensure_residue_batch_dim(peptide_mask)
    central_dscore_mask = _ensure_residue_batch_dim(central_dscore_mask)
    anchor_class = anchor_class.reshape((-1,))

    consecutive = ((residue_index[:, 1:] - residue_index[:, :-1]) == 1).astype(jnp.float32)

    phi_pred_sin, phi_pred_cos = _dihedral_sin_cos(
        pred_pos[:, :-1, ATOM_C, :],
        pred_pos[:, 1:, ATOM_N, :],
        pred_pos[:, 1:, ATOM_CA, :],
        pred_pos[:, 1:, ATOM_C, :])
    phi_native_sin, phi_native_cos = _dihedral_sin_cos(
        native_pos[:, :-1, ATOM_C, :],
        native_pos[:, 1:, ATOM_N, :],
        native_pos[:, 1:, ATOM_CA, :],
        native_pos[:, 1:, ATOM_C, :])
    phi_mask = (
        central_dscore_mask[:, 1:] * peptide_mask[:, :-1] * peptide_mask[:, 1:] * consecutive *
        pred_mask[:, :-1, ATOM_C] * pred_mask[:, 1:, ATOM_N] *
        pred_mask[:, 1:, ATOM_CA] * pred_mask[:, 1:, ATOM_C] *
        native_mask[:, :-1, ATOM_C] * native_mask[:, 1:, ATOM_N] *
        native_mask[:, 1:, ATOM_CA] * native_mask[:, 1:, ATOM_C])

    psi_pred_sin, psi_pred_cos = _dihedral_sin_cos(
        pred_pos[:, :-1, ATOM_N, :],
        pred_pos[:, :-1, ATOM_CA, :],
        pred_pos[:, :-1, ATOM_C, :],
        pred_pos[:, 1:, ATOM_N, :])
    psi_native_sin, psi_native_cos = _dihedral_sin_cos(
        native_pos[:, :-1, ATOM_N, :],
        native_pos[:, :-1, ATOM_CA, :],
        native_pos[:, :-1, ATOM_C, :],
        native_pos[:, 1:, ATOM_N, :])
    psi_mask = (
        central_dscore_mask[:, :-1] * peptide_mask[:, :-1] * peptide_mask[:, 1:] * consecutive *
        pred_mask[:, :-1, ATOM_N] * pred_mask[:, :-1, ATOM_CA] *
        pred_mask[:, :-1, ATOM_C] * pred_mask[:, 1:, ATOM_N] *
        native_mask[:, :-1, ATOM_N] * native_mask[:, :-1, ATOM_CA] *
        native_mask[:, :-1, ATOM_C] * native_mask[:, 1:, ATOM_N])

    phi_dist = _angle_dscore_sin_cos(
        phi_pred_sin, phi_pred_cos,
        phi_native_sin, phi_native_cos)
    psi_dist = _angle_dscore_sin_cos(
        psi_pred_sin, psi_pred_cos,
        psi_native_sin, psi_native_cos)

    # Do not rely on NaN * 0 masking; JAX follows IEEE behavior, so NaN * 0 is
    # still NaN. Replace invalid/unselected positions before summing.
    phi_dist = jnp.where(phi_mask > 0, phi_dist, 0.0)
    psi_dist = jnp.where(psi_mask > 0, psi_dist, 0.0)
    phi_dist = jnp.nan_to_num(phi_dist, nan=0.0, posinf=0.0, neginf=0.0)
    psi_dist = jnp.nan_to_num(psi_dist, nan=0.0, posinf=0.0, neginf=0.0)

    angle_dist_sum = jnp.sum(phi_dist, axis=-1) + jnp.sum(psi_dist, axis=-1)
    angle_count = jnp.sum(phi_mask, axis=-1) + jnp.sum(psi_mask, axis=-1)
    # Original HLA3DB normalization uses num_angles = anchor_class * 2 - 6,
    # not the observed mask count. angle_count is retained only to skip invalid
    # examples where the peptide/central residues are missing after cropping.
    num_angles = anchor_class * 2.0 - 6.0
    valid = ((angle_count > 0) & (num_angles > 0)).astype(jnp.float32)
    scaled_dscore = angle_dist_sum * 8.0 / jnp.maximum(num_angles, 1.0)
    #debug
    #jax.debug.print("pred_pos finite: {x}", x=jnp.all(jnp.isfinite(pred_pos)))
    #jax.debug.print("native_pos finite: {x}", x=jnp.all(jnp.isfinite(native_pos)))
    #jax.debug.print("phi_pred_sincos finite: {x}", x=(jnp.all(jnp.isfinite(phi_pred_sin)) & jnp.all(jnp.isfinite(phi_pred_cos))))
    #jax.debug.print("phi_native_sincos finite: {x}", x=(jnp.all(jnp.isfinite(phi_native_sin)) & jnp.all(jnp.isfinite(phi_native_cos))))
    #jax.debug.print("psi_pred_sincos finite: {x}", x=(jnp.all(jnp.isfinite(psi_pred_sin)) & jnp.all(jnp.isfinite(psi_pred_cos))))
    #jax.debug.print("psi_native_sincos finite: {x}", x=(jnp.all(jnp.isfinite(psi_native_sin)) & jnp.all(jnp.isfinite(psi_native_cos))))
    #jax.debug.print("angle_count={x}, num_angles={y}", x=angle_count, y=num_angles)

    #jax.debug.print("phi_mask sum={x}", x=jnp.sum(phi_mask, axis=-1))
    #jax.debug.print("psi_mask sum={x}", x=jnp.sum(psi_mask, axis=-1))
    #jax.debug.print("central mask sum={x}", x=jnp.sum(central_dscore_mask, axis=-1))
    #jax.debug.print("peptide mask sum={x}", x=jnp.sum(peptide_mask, axis=-1))
    #jax.debug.print("anchor_class={x}", x=anchor_class)
    #debug
    return jnp.sum(scaled_dscore * valid) / jnp.maximum(jnp.sum(valid), 1.0)

def get_loss_fn(model_params, key, processed_feature_dict, structure_flag):
    """Compute AlphaFold structure loss plus peptide D-score loss.

    Binder labels, binder class features, and PAE/pLDDT classifier
    outputs are not used in this AlphaFold-only fine-tuning version.
    """
    peptide_mask = processed_feature_dict.pop('peptide_mask')
    central_dscore_mask = processed_feature_dict.pop('central_dscore_mask')
    anchor_class = processed_feature_dict.pop('anchor_class')
    # Remove non-AlphaFold bookkeeping fields if they are present in older TSV/loaders.
    for extra_key in [
        'binder_class_1hot', 'partner1_mask', 'partner2_mask',
        'labels', 'native_exists'
    ]:
        processed_feature_dict.pop(extra_key, None)

    predicted_dict, af_loss = model_runner.apply(
        model_params, key, processed_feature_dict)

    fake_native_weight = jnp.array(FLAGS.fake_native_weight, jnp.float32)
    dscore_weight = jnp.array(FLAGS.dscore_weight, jnp.float32)
    sf = jnp.array(structure_flag, jnp.float32)  # 1.0 for real native, 0.0 for predicted/fake native
    not_sf = jnp.ones(sf.shape, dtype=sf.dtype) - sf
    structure_weight = sf + not_sf * fake_native_weight

    # Same structure-loss weighting logic as the original script, but no binder loss.
    # Important: when dscore_weight is exactly zero, do not call peptide_dscore_loss().
    # This avoids tracing its gradient path during AF-only control runs.
    if FLAGS.dscore_weight == 0.0:
        dscore_loss = jnp.array(0.0, jnp.float32)
        loss = structure_weight * af_loss[0]
    else:
        dscore_loss = peptide_dscore_loss(
            predicted_dict, processed_feature_dict,
            peptide_mask, central_dscore_mask, anchor_class)
        loss = structure_weight * af_loss[0] + dscore_weight * dscore_loss
        #loss = structure_weight * af_loss[0]
    #debug
    jax.debug.print(
        "af={af}, dscore={ds}, weighted_dscore={wds}, total={total}",
        af=af_loss[0], ds=dscore_loss, wds=dscore_weight * dscore_loss, total=loss,)
    #debug
    return loss, (predicted_dict, dscore_loss)

def train_step(model_params, key, batch, structure_flag):
    (loss, (predicted_dict, dscore_loss)), grads = jax.value_and_grad(
        get_loss_fn, has_aux=True)(model_params, key, batch, structure_flag)
    grads = jax.lax.pmean(grads, axis_name='model_ax')
    loss = jax.lax.pmean(loss, axis_name='model_ax')
    dscore_loss = jax.lax.pmean(dscore_loss, axis_name='model_ax')
    return loss, grads, predicted_dict, dscore_loss

def norm_grads_per_example(grads, l2_norm_clip=0.1):
    nonempty_grads, tree_def = jax.tree_util.tree_flatten(grads)
    #total_grad_norm = jnp.linalg.norm([jnp.linalg.norm(neg.ravel()) for neg in nonempty_grads])
    total_grad_norm = jnp.sqrt(
    sum(jnp.sum(neg ** 2) for neg in nonempty_grads)
    )
    divisor = jnp.maximum(total_grad_norm / l2_norm_clip, 1.)
    normalized_nonempty_grads = [g / divisor for g in nonempty_grads]
    grads = jax.tree_util.tree_unflatten(tree_def, normalized_nonempty_grads)
    return grads


def _get_peptide_positions_from_row(row):
    """Return sorted 0-indexed full-sequence peptide positions.

    If the dataset has peptide_positions, use it. Otherwise assume the final
    slash-separated chain in target_chainseq is the peptide.
    """
    if hasattr(row, 'peptide_positions') and not pd.isna(row.peptide_positions):
        return sorted({int(x) for x in str(row.peptide_positions).split(';') if x})

    chain_lengths = [len(x) for x in row.target_chainseq.split('/')]
    peptide_start = sum(chain_lengths[:-1])
    return list(range(peptide_start, peptide_start + chain_lengths[-1]))


_ANCHOR_INFO_BY_PDBID = None


def _load_anchor_info_by_pdbid():
    """Load anchor_class.csv once and return a pdbid -> row dictionary."""
    global _ANCHOR_INFO_BY_PDBID
    if _ANCHOR_INFO_BY_PDBID is not None:
        return _ANCHOR_INFO_BY_PDBID

    anchor_info = {}
    anchor_file = FLAGS.anchor_class_file
    if anchor_file and os.path.exists(anchor_file):
        df = pd.read_csv(anchor_file)
    elif anchor_file and os.path.exists(os.path.join(os.getcwd(), anchor_file)):
        df = pd.read_csv(os.path.join(os.getcwd(), anchor_file))
    else:
        print(f'WARNING: anchor_class_file not found: {anchor_file}; falling back to dataset/default anchor_class')
        _ANCHOR_INFO_BY_PDBID = anchor_info
        return anchor_info

    required = {'pdbid', 'anchor_class', 'anchor1', 'anchor2'}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f'anchor_class_file is missing required columns: {sorted(missing)}')

    for _, r in df.iterrows():
        pdbid = str(r.pdbid).strip()
        item = {
            'anchor_class': int(r.anchor_class),
            'anchor1': int(r.anchor1),
            'anchor2': int(r.anchor2),
        }
        anchor_info[pdbid] = item
        # Also allow matching PDB IDs with chain suffixes like 1ABC-A.
        anchor_info[pdbid.split('-')[0]] = item

    print(f'Loaded {len(df)} rows from anchor_class_file: {anchor_file}')
    _ANCHOR_INFO_BY_PDBID = anchor_info
    return anchor_info


def _row_pdbid_candidates(row):
    """PDB identifiers to try when looking up anchor_class.csv."""
    candidates = []
    for col in ['pdbid', 'targetid', 'native_pdbid']:
        if hasattr(row, col):
            value = getattr(row, col)
            if not pd.isna(value):
                value = str(value).strip()
                candidates.extend([value, value.split('-')[0]])
    # De-duplicate while preserving order.
    out = []
    seen = set()
    for c in candidates:
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _anchor_info_from_row(row, peptide_len, peptide_start_0):
    """Return (anchor_class, anchor1_local, anchor2_local).

    anchor_class.csv stores anchors in HLA3DB 1-based full-complex numbering
    such as 182/189 for P2/P9 when the HLA chain has 180 residues. This
    function converts those to peptide-local 1-based positions. If dataset
    columns anchor_class/anchor1/anchor2 are already present, they are used as a
    fallback.
    """
    anchor_info = _load_anchor_info_by_pdbid()
    info = None
    for pdbid in _row_pdbid_candidates(row):
        if pdbid in anchor_info:
            info = anchor_info[pdbid]
            break

    if info is not None:
        anchor_class = int(info['anchor_class'])
        raw_anchor1 = int(info['anchor1'])
        raw_anchor2 = int(info['anchor2'])
    else:
        if hasattr(row, 'anchor_class') and not pd.isna(row.anchor_class):
            anchor_class = int(row.anchor_class)
        else:
            anchor_class = int(FLAGS.default_anchor_class)

        if hasattr(row, 'anchor1') and hasattr(row, 'anchor2') and not pd.isna(row.anchor1) and not pd.isna(row.anchor2):
            raw_anchor1 = int(row.anchor1)
            raw_anchor2 = int(row.anchor2)
        else:
            # Common 9-mer HLA case: anchor_class 7 -> P2/P9.
            raw_anchor2 = peptide_len
            raw_anchor1 = max(1, raw_anchor2 - anchor_class)

    def to_peptide_local(raw_anchor):
        # Already peptide-local 1-based.
        if 1 <= raw_anchor <= peptide_len:
            return raw_anchor
        # HLA3DB/full-complex 1-based numbering. peptide_start_0 is the
        # 0-based full target position of peptide P1, so raw - peptide_start_0
        # maps 182 -> 2 when peptide_start_0 is 180.
        return raw_anchor - peptide_start_0

    anchor1 = to_peptide_local(raw_anchor1)
    anchor2 = to_peptide_local(raw_anchor2)

    if not (1 <= anchor1 <= peptide_len and 1 <= anchor2 <= peptide_len and anchor1 < anchor2):
        raise ValueError(
            f'Invalid anchor positions for row {getattr(row, "pdbid", getattr(row, "targetid", "unknown"))}: '
            f'raw=({raw_anchor1}, {raw_anchor2}), local=({anchor1}, {anchor2}), peptide_len={peptide_len}'
        )

    return anchor_class, anchor1, anchor2


def _peptide_and_central_dscore_masks_from_row(row, target_trim_positions):
    """Create crop-sized peptide and HLA3DB central-D-score masks.

    The central mask selects peptide-local positions anchor1+2 through
    anchor2-2, exactly like common.py/read_central_dihedral(). Both phi and psi
    for these residues are used by peptide_dscore_loss().
    """
    peptide_positions = _get_peptide_positions_from_row(row)
    peptide_set = set(peptide_positions)
    full_to_pep_pos = {full_pos: i + 1 for i, full_pos in enumerate(peptide_positions)}
    peptide_start_0 = min(peptide_positions)
    anchor_class, anchor1, anchor2 = _anchor_info_from_row(row, len(peptide_positions), peptide_start_0)
    center_start_pos = anchor1 + 2
    center_end_pos = anchor2 - 2

    peptide_mask = np.zeros((FLAGS.crop_size,), np.float32)
    central_mask = np.zeros((FLAGS.crop_size,), np.float32)
    for trim_pos, full_pos in enumerate(target_trim_positions):
        if trim_pos >= FLAGS.crop_size:
            break
        if full_pos in peptide_set:
            peptide_mask[trim_pos] = 1.0
            pep_pos = full_to_pep_pos[full_pos]
            if center_start_pos <= pep_pos <= center_end_pos:
                central_mask[trim_pos] = 1.0

    return peptide_mask, central_mask, np.array(anchor_class, np.float32)


def _dscore_peptide_positions_from_row(row, target_trim_positions):
    """Return cropped residue indices for the peptide used by validation D-score."""
    peptide_positions = _get_peptide_positions_from_row(row)
    trim_to_crop = {int(orig): i for i, orig in enumerate(target_trim_positions)}
    return np.array([trim_to_crop.get(int(pos), -1) for pos in peptide_positions], dtype=np.int32)


def create_batch_from_dataset_row(row, training=True):
    ''' row is a pandas Series that has fields:

    required_fields:
      target_chainseq  ('/'-separated chain amino acid sequences)
      templates_alignfile
      native_pdbfile
      native_alignstring (';'-separated intpairs eg '0:0;1:1;3:2;4:3')
      native_exists (True or False or 0 or 1)

    optional fields:
      target_trim_positions (used to subset the target for memory reasons)
      peptide_positions (0-indexed target positions; defaults to final chain)
      native_identities (for debugging PDB IO)
      native_len (for debugging PDB IO)

    '''

    debug = FLAGS.debug
    verbose = FLAGS.verbose
    if True: #debug:
        print('create_batch_from_dataset_row:',
              getattr(row,'targetid','unk'),
              getattr(row,'pdbid','unk'))

    nres = len(row.target_chainseq.replace('/',''))
    if hasattr(row, 'target_trim_positions'):
        target_trim_positions = [int(x) for x in row.target_trim_positions.split(';')]
        if target_trim_positions != list(range(nres)):
            print('WARNING: target_trimming:',
                  [x for x in range(nres) if x not in target_trim_positions])
    else:
        target_trim_positions = list(range(nres))
    native_align = {int(x.split(':')[0]):int(x.split(':')[1])
                    for x in row.native_alignstring.split(';')}

    native_identities = getattr(row, 'native_identities', None)
    native_identities = None if pd.isna(native_identities) else native_identities
    native_len = getattr(row, 'native_len', None)
    native_len = None if pd.isna(native_len) else native_len

    # Keep training stochastic unless --no_random is enabled, but always
    # use a fixed preprocessing seed for validation.
    if training:
        random_seed = 0 if FLAGS.no_random else None
    else:
        random_seed = 0

    batch = predict_utils.create_batch_for_training(
        row.target_chainseq, target_trim_positions, row.templates_alignfile,
        row.native_pdbfile, native_align, FLAGS.crop_size, model_runner,
        native_identities = native_identities,
        native_len = native_len,
        debug=debug,
        verbose=verbose,
        random_seed=random_seed,
    )

    # AlphaFold-only fine-tuning: no binder labels, binder classes, peptide masks,
    # partner masks, or classifier inputs are created. peptide_mask is used only
    # for the D-score structural loss.
    batch['peptide_mask'], batch['central_dscore_mask'], batch['anchor_class'] = _peptide_and_central_dscore_masks_from_row(row, target_trim_positions)
    batch['native_exists'] = bool(row.native_exists)

    # Validation-only metadata for D-score reporting and predicted PDB output.
    batch['dscore_pep_positions'] = _dscore_peptide_positions_from_row(row, target_trim_positions)
    batch['targetid'] = str(getattr(row, 'targetid', getattr(row, 'pdbid', 'unk')))
    batch['native_pdbfile'] = str(getattr(row, 'native_pdbfile', ''))

    return batch




class CustomPandasDataset(torch.utils.data.Dataset):
    def __init__(self, df, loader):
        self.df = df.copy()
        self.loader = loader
    def __len__(self):
        return self.df.shape[0]
    def __getitem__(self, index):
        if FLAGS.debug:
            print('CustomPandasDataset:get_item: index=', index)
        out = self.loader(self.df.iloc[index])
        return out

def collate(samples):
    out_dict = {}
    for name in train_utils.list_a + train_utils.list_a_templates:
        values = [item[name][0,] for item in samples]
        out_dict[name] = np.stack(values, axis=0)
    for name in train_utils.list_b + train_utils.list_b_templates:
        values = [item[name] for item in samples]
        out_dict[name] = np.stack(values, axis=0)
    aatype_ = [item['aatype'][0,] for item in samples]
    out_dict['aatype_'] = np.stack(aatype_, axis=0)
    out_dict['peptide_mask'] = np.stack([item['peptide_mask'] for item in samples], axis=0)
    out_dict['central_dscore_mask'] = np.stack([item['central_dscore_mask'] for item in samples], axis=0)
    out_dict['anchor_class'] = np.stack([item['anchor_class'] for item in samples], axis=0)
    out_dict['native_exists'] = [item['native_exists'] for item in samples]
    out_dict['dscore_pep_positions'] = np.stack([item['dscore_pep_positions'] for item in samples], axis=0)
    out_dict['targetid'] = [item.get('targetid', 'unk') for item in samples]
    out_dict['native_pdbfile'] = [item.get('native_pdbfile', '') for item in samples]
    return out_dict


def prep_batch_for_step(batch, training):
    torsion_dict = atom37_to_torsion_angles(
        jnp.array(batch['aatype_']),
        jnp.array(batch['all_atom_positions']),
        jnp.array(batch['all_atom_mask']))
    batch['chi_mask'] = torsion_dict['torsion_angles_mask'][:,:,3:] #[B, N, 4] for 4 chi
    sin_chi = torsion_dict['torsion_angles_sin_cos'][:,:,3:,0]
    cos_chi = torsion_dict['torsion_angles_sin_cos'][:,:,3:,1]
    batch['chi_angles'] = jnp.arctan2(sin_chi, cos_chi) #[B, N, 4] for 4 chi angles
    rigidgroups_dict = atom37_to_frames(
        jnp.array(batch['aatype_']),
        jnp.array(batch['all_atom_positions']),
        jnp.array(batch['all_atom_mask']))
    batch.update(rigidgroups_dict)
    for key_, value_ in batch.items(): # add the 'ensemble' dimension??
        if key_ in train_utils.list_a + train_utils.list_a_templates:
            batch[key_] = value_[:,None,]
        if key_ in train_utils.list_c:
            batch[key_] = value_[:,None,]
    for item_ in train_utils.pdb_key_list_int:
        batch[item_] = jnp.array(batch[item_], jnp.int32)
    batch['peptide_mask'] = jnp.array(batch['peptide_mask'], jnp.float32)
    batch['central_dscore_mask'] = jnp.array(batch['central_dscore_mask'], jnp.float32)
    batch['anchor_class'] = jnp.array(batch['anchor_class'], jnp.float32)
    if training:
        if FLAGS.random_recycling:
            batch['num_iter_recycling'] = jnp.array(np.tile(
                np.random.randint(0, model_config.model.num_recycle+1, 1)[None,],
                (batch_size, model_config.model.num_recycle)), jnp.int32)
            print('num_iter_recycling:', batch['num_iter_recycling'])
        else:
            print('not setting num_iter_recycling!!! will do',
                  model_config.model.num_recycle,'recycles')

    batch.pop('dscore_pep_positions', None) # validation-only metadata
    batch.pop('targetid', None)              # validation-output metadata
    batch.pop('native_pdbfile', None)        # validation-output metadata
    del batch['native_exists'] # already got this
    if FLAGS.verbose:
        print('prep_batch_for_step_final_features:', ' '.join(batch.keys()))



def _remove_leading_singleton_axes(array, final_rank):
    """Remove pmap/batch/ensemble singleton axes until array has final_rank dims."""
    array = np.asarray(array)
    while array.ndim > final_rank:
        array = array[0]
    return array


def protein_from_prediction(batch, predicted_dict, b_factors=None):
    """Build an AlphaFold Protein object from the current model prediction."""
    fold_output = predicted_dict['structure_module']
    atom_positions = _remove_leading_singleton_axes(fold_output['final_atom_positions'], 3)
    atom_mask = _remove_leading_singleton_axes(fold_output['final_atom_mask'], 2)
    aatype = _remove_leading_singleton_axes(batch['aatype'], 1)
    residue_index = _remove_leading_singleton_axes(batch['residue_index'], 1) + 1

    if b_factors is None:
        b_factors = np.zeros_like(atom_mask)
    else:
        b_factors = _remove_leading_singleton_axes(b_factors, 2)

    return Protein(
        aatype=aatype,
        atom_positions=atom_positions,
        atom_mask=atom_mask,
        residue_index=residue_index,
        b_factors=b_factors)


def _safe_pdb_basename(target_id, native_pdbfile='', fallback='validation_prediction'):
    """Return a filesystem-safe PDB basename, preferring targetid/pdbid metadata."""
    name = str(target_id) if target_id is not None else ''
    if name in ['', 'nan', 'None', 'unk'] and native_pdbfile:
        name = os.path.basename(str(native_pdbfile))
        if name.endswith('.pdb'):
            name = name[:-4]
        if name.endswith('_reordered'):
            name = name[:-10]
    if name in ['', 'nan', 'None', 'unk']:
        name = fallback
    return ''.join(ch if ch.isalnum() or ch in ['-', '_', '.'] else '_' for ch in name)


def dump_validation_prediction_pdb(batch, predicted_dict, target_id, native_pdbfile, epoch, batch_index):
    """Write one validation predicted structure as <targetid><suffix>.pdb."""
    os.makedirs(FLAGS.valid_pdb_dir, exist_ok=True)
    base = _safe_pdb_basename(
        target_id,
        native_pdbfile,
        fallback=f'valid_epoch{epoch:02d}_batch{batch_index:04d}')
    outfile = os.path.join(FLAGS.valid_pdb_dir, f'{base}{FLAGS.valid_pdb_suffix}.pdb')
    unrelaxed_protein = protein_from_prediction(batch, predicted_dict)
    with open(outfile, 'w') as f:
        f.write(protein.to_pdb(unrelaxed_protein))
    print('made validation pdb:', outfile)
    return outfile


def _dihedral_degrees(p0, p1, p2, p3):
    """Dihedral angle in degrees for four 3D points."""
    b0 = -(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2
    norm_b1 = np.linalg.norm(b1)
    if norm_b1 < 1e-6:
        return np.nan
    b1 = b1 / norm_b1
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    if np.linalg.norm(v) < 1e-6 or np.linalg.norm(w) < 1e-6:
        return np.nan
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return float(np.degrees(np.arctan2(y, x)))


def _cyclic_angle_distance(angle1, angle2):
    return 2.0 * (1.0 - np.cos(np.radians(angle1) - np.radians(angle2)))


def _single_structure_central_phi_psi(atom_positions, atom_mask, peptide_mask, central_dscore_mask, residue_index):
    """Return central phi/psi angles using the same masks as peptide_dscore_loss().

    This is anchor-aware: central_dscore_mask should already select peptide-local
    residues anchor1+2 through anchor2-2. For each selected central residue, this
    function attempts to compute phi and psi only when the neighboring peptide
    residue is present, residue_index is consecutive, and all required backbone
    atoms are present. This mirrors the masking logic in peptide_dscore_loss().
    """
    n_idx = residue_constants.atom_order['N']
    ca_idx = residue_constants.atom_order['CA']
    c_idx = residue_constants.atom_order['C']

    atom_positions = np.asarray(atom_positions)
    atom_mask = np.asarray(atom_mask)
    peptide_mask = np.asarray(peptide_mask, dtype=np.float32)
    central_dscore_mask = np.asarray(central_dscore_mask, dtype=np.float32)
    residue_index = np.asarray(residue_index)

    while peptide_mask.ndim > 1:
        peptide_mask = peptide_mask[0]
    while central_dscore_mask.ndim > 1:
        central_dscore_mask = central_dscore_mask[0]
    while residue_index.ndim > 1:
        residue_index = residue_index[0]

    angles = []
    nres = atom_positions.shape[0]
    for res in range(nres):
        if central_dscore_mask[res] < 0.5 or peptide_mask[res] < 0.5:
            continue

        # phi_i = dihedral(C_{i-1}, N_i, CA_i, C_i)
        prev_res = res - 1
        if prev_res >= 0:
            consecutive_prev = (residue_index[res] - residue_index[prev_res]) == 1
            needed_phi = [(prev_res, c_idx), (res, n_idx), (res, ca_idx), (res, c_idx)]
            if (consecutive_prev and peptide_mask[prev_res] >= 0.5 and
                    all(atom_mask[r, a] >= 0.5 for r, a in needed_phi)):
                phi = _dihedral_degrees(atom_positions[prev_res, c_idx],
                                        atom_positions[res, n_idx],
                                        atom_positions[res, ca_idx],
                                        atom_positions[res, c_idx])
                if not np.isnan(phi):
                    angles.append(('phi', res, phi))

        # psi_i = dihedral(N_i, CA_i, C_i, N_{i+1})
        next_res = res + 1
        if next_res < nres:
            consecutive_next = (residue_index[next_res] - residue_index[res]) == 1
            needed_psi = [(res, n_idx), (res, ca_idx), (res, c_idx), (next_res, n_idx)]
            if (consecutive_next and peptide_mask[next_res] >= 0.5 and
                    all(atom_mask[r, a] >= 0.5 for r, a in needed_psi)):
                psi = _dihedral_degrees(atom_positions[res, n_idx],
                                        atom_positions[res, ca_idx],
                                        atom_positions[res, c_idx],
                                        atom_positions[next_res, n_idx])
                if not np.isnan(psi):
                    angles.append(('psi', res, psi))

    return angles


def compute_validation_dscore_from_batch(batch, predicted_dict):
    """Scaled central D-score matching peptide_dscore_loss().

    Uses central_dscore_mask rather than a hard-coded P4-P7 selection, and rescales
    the raw sum as: dscore_scaled = dscore * 14 / (anchor_class * 2 - 6).
    """
    pred_pos = np.asarray(predicted_dict['structure_module']['final_atom_positions'])
    pred_mask = np.asarray(predicted_dict['structure_module']['final_atom_mask'])
    native_pos = np.asarray(batch['all_atom_positions'])
    native_mask = np.asarray(batch['all_atom_mask'])
    peptide_mask = np.asarray(batch['peptide_mask'])
    central_dscore_mask = np.asarray(batch['central_dscore_mask'])
    anchor_class = np.asarray(batch['anchor_class'])
    residue_index = np.asarray(batch['residue_index'])

    while pred_pos.ndim > 3:
        pred_pos = pred_pos[0]
        pred_mask = pred_mask[0]
    while native_pos.ndim > 3:
        native_pos = native_pos[0]
        native_mask = native_mask[0]
    while peptide_mask.ndim > 1:
        peptide_mask = peptide_mask[0]
    while central_dscore_mask.ndim > 1:
        central_dscore_mask = central_dscore_mask[0]
    while residue_index.ndim > 1:
        residue_index = residue_index[0]

    anchor_class = float(np.asarray(anchor_class).reshape(-1)[0])
    num_angles = anchor_class * 2.0 - 6.0
    if num_angles <= 0:
        return np.nan

    native_angles = _single_structure_central_phi_psi(
        native_pos, native_mask, peptide_mask, central_dscore_mask, residue_index)
    pred_angles = _single_structure_central_phi_psi(
        pred_pos, pred_mask, peptide_mask, central_dscore_mask, residue_index)

    native_by_key = {(kind, res): angle for kind, res, angle in native_angles}
    pred_by_key = {(kind, res): angle for kind, res, angle in pred_angles}
    common_keys = sorted(set(native_by_key).intersection(pred_by_key), key=lambda x: (x[1], x[0]))
    if len(common_keys) == 0:
        return np.nan

    angle_dist_sum = sum(_cyclic_angle_distance(pred_by_key[k], native_by_key[k]) for k in common_keys)
    scaled_dscore = angle_dist_sum * 8.0 / max(num_angles, 1.0)
    return float(scaled_dscore)

def show(name, thing):
    print(name, end=': ')
    if hasattr(thing, 'items'):
        for k,v in thing.items():
            show(k, v)
    elif hasattr(thing, 'shape'):
        print('shape=', thing.shape, end=', ')

def compute_valid_stats(valid_loader, replicated_params, jax_key):
    if FLAGS.no_valid:
        print('no valid stats')
        return
    temp_train_loss = []
    temp_dscore_loss = []
    temp_lddt_ca = []
    temp_distogram = []
    temp_masked_msa = []
    temp_pred_lddt = []
    temp_chi_loss = []
    temp_fape = []
    temp_sidechain_fape = []
    temp_valid_dscore = []
    temp_valid_dscore_similar = []
    for n, batch in enumerate(valid_loader):
        print('test_epoch:', e, 'batch:', n)
        structure_flag = batch['native_exists'][0]
        valid_target_id = batch.get('targetid', [f'valid_epoch{e:02d}_batch{n:04d}'])[0]
        valid_native_pdbfile = batch.get('native_pdbfile', [''])[0]
        prep_batch_for_step(batch, False)  # keep original behavior/change to false
        # Deterministic validation RNG: each validation batch gets a stable,
        # reproducible key derived from the fixed base seed and batch index.
        subkey = jax.random.fold_in(jax.random.PRNGKey(0), n)

        loss, grads, predicted_dict, dscore_loss = jax.pmap(
            train_step, in_axes=(0, None, 0, None), axis_name='model_ax'
        )(replicated_params, subkey, batch, structure_flag)

        print('test_epoch_n=', e, n, 'loss=', loss[0],
              'dscore_loss=', dscore_loss[0],
              'lddt_ca=', np.mean(predicted_dict['predicted_lddt']['lddt_ca']),
              'fape=', np.mean(predicted_dict['structure_module']['fape']))

        if FLAGS.dump_valid_pdbs:
            dump_validation_prediction_pdb(
                batch, predicted_dict, valid_target_id, valid_native_pdbfile, e, n)

        temp_train_loss.append(np.mean(loss[0]))
        temp_dscore_loss.append(np.mean(dscore_loss[0]))
        temp_lddt_ca.append(np.mean(predicted_dict['predicted_lddt']['lddt_ca']))
        temp_distogram.append(np.mean(predicted_dict['distogram']['loss']))
        temp_masked_msa.append(np.mean(predicted_dict['masked_msa']['loss']))
        temp_pred_lddt.append(np.mean(predicted_dict['predicted_lddt']['loss']))
        temp_chi_loss.append(np.mean(predicted_dict['structure_module']['chi_loss']))
        temp_fape.append(np.mean(predicted_dict['structure_module']['fape']))
        temp_sidechain_fape.append(np.mean(predicted_dict['structure_module']['sidechain_fape']))
        if FLAGS.calc_valid_dscore and bool(structure_flag):
            valid_dscore = compute_validation_dscore_from_batch(batch, predicted_dict)
            if not np.isnan(valid_dscore):
                temp_valid_dscore.append(valid_dscore)
                temp_valid_dscore_similar.append(float(valid_dscore <= FLAGS.dscore_similarity_cutoff))
            print('test_epoch_n=', e, n, 'valid_dscore=', valid_dscore,
                  'valid_dscore_similar=', (False if np.isnan(valid_dscore) else valid_dscore <= FLAGS.dscore_similarity_cutoff))
        mean_loss = round(float(np.mean(temp_train_loss)), 4)
        mean_dscore_loss = round(float(np.mean(temp_dscore_loss)), 4)
        lddt_ca = round(float(np.mean(temp_lddt_ca)), 4)
        distogram = round(float(np.mean(temp_distogram)), 4)
        masked_msa = round(float(np.mean(temp_masked_msa)), 4)
        fape = round(float(np.mean(temp_fape)), 4)
        sidechain_fape = round(float(np.mean(temp_sidechain_fape)), 4)
        chi_loss = round(float(np.mean(temp_chi_loss)), 4)
        valid_dscore_mean = 'nan' if len(temp_valid_dscore) == 0 else round(float(np.mean(temp_valid_dscore)), 4)
        valid_dscore_sim = 'nan' if len(temp_valid_dscore_similar) == 0 else round(float(np.mean(temp_valid_dscore_similar)), 4)
        print(f'test_Step: {global_step} {n}, loss: {mean_loss}, dscore_loss: {mean_dscore_loss}, valid_dscore: {valid_dscore_mean}, valid_dscore_similar_frac: {valid_dscore_sim}, lddt: {lddt_ca}, fape: {fape}, sc_fape: {sidechain_fape}, chi: {chi_loss}, disto: {distogram}, msa: {masked_msa}', flush=True)
        sys.stdout.flush()


######################################################################################88
######################################################################################88
######################################################################################88
######################################################################################88
######################################################################################88
######################################################################################88
######################################################################################88


train_df = pd.read_table(FLAGS.train_dataset)
valid_df = pd.read_table(FLAGS.valid_dataset)

def create_train_batch(row):
    return create_batch_from_dataset_row(row, training=True)


def create_valid_batch(row):
    return create_batch_from_dataset_row(row, training=False)


training_set = CustomPandasDataset(train_df, create_train_batch)
valid_set = CustomPandasDataset(valid_df, create_valid_batch)

common_loader_args = {
    'num_workers': FLAGS.num_cpus,
    'pin_memory': False,
    'batch_size': batch_size,
    'collate_fn': collate,
}

train_loader = torch.utils.data.DataLoader(
    training_set,
    shuffle=True,
    **common_loader_args,
)

valid_loader = torch.utils.data.DataLoader(
    valid_set,
    shuffle=False,
    **common_loader_args,
)

if FLAGS.no_ramp:
    scheduler = optax.linear_schedule(1e-3, 1e-3, 1000, 0)
else:
    scheduler = optax.linear_schedule(0.0, 1e-3, 1000, 0)

# Combining gradient transforms using `optax.chain`.
chain_me = [
    optax.scale_by_adam(b1=0.9, b2=0.999, eps=1e-6),  # Use the updates from adam.
    optax.scale_by_schedule(scheduler),  # Use the learning rate from the scheduler.
    # Scale updates by -1 since optax.apply_updates is additive and we want to descend on the loss.
    optax.scale(-1.0*FLAGS.lr_coef),
]

gradient_transform = optax.chain(*chain_me)

n_devices = jax.local_device_count()
assert n_devices == 1, 'Not tested with multiple devices yet'
replicated_params = jax.tree_map(lambda x: jnp.array([x] * n_devices), model_params)

opt_state = gradient_transform.init(replicated_params)
global_step = 0

checkpoint_dir = os.path.join(FLAGS.outprefix + "_checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)

if FLAGS.test_load:
    # test the setup
    for n, batch in enumerate(train_loader):
        print('trainload:',n)
        sys.stdout.flush()
    for n, batch in enumerate(valid_loader):
        print('validload:',n)
        sys.stdout.flush()

if FLAGS.notrain:
    num_epochs = 1
else:
    num_epochs = FLAGS.num_epochs

for e in range(num_epochs):
    temp_train_loss = []
    temp_dscore_loss = []
    temp_lddt_ca = []
    temp_distogram = []
    temp_masked_msa = []
    temp_exper_res = []
    temp_pred_lddt = []
    temp_chi_loss = []
    temp_fape = []
    temp_sidechain_fape = []
    grads_sum, grads_sum_count = None, 0

    for n, batch in enumerate(train_loader):
        if FLAGS.notrain:
            break
        structure_flag = batch['native_exists'][0]
        print('train_epoch:', e, 'batch:', n)
        prep_batch_for_step(batch, True)

        if FLAGS.verbose:
            for key_, value_ in batch.items():
                print('before_train_step:', key_, type(value_),
                      getattr(value_, 'shape', 'unk_shape'),
                      getattr(value_, 'dtype', 'unk_dtype'))
        sys.stdout.flush()


        #quoting from the docs: "Like with vmap, we can use in_axes to specify whether an argument to the parallelized function should be broadcast (None), or whether it should be split along a given axis. Note, however, that unlike vmap, only the leading axis (0) is supported by pmap"

        if FLAGS.no_random:
            subkey = jax.random.PRNGKey(0)
        else:
            jax_key, subkey = jax.random.split(jax_key)

        loss, grads, predicted_dict, dscore_loss = jax.pmap(
            train_step, in_axes=(0, None, 0, None), axis_name='model_ax'
        )(replicated_params, subkey, batch, structure_flag)


        print('train_epoch_n=', e, n, 'loss=', loss[0],
              'dscore_loss=', dscore_loss[0],
              'structure_flag:', structure_flag,
              'lddt_ca=', np.mean(predicted_dict['predicted_lddt']['lddt_ca']),
              'fape=', np.mean(predicted_dict['structure_module']['fape']),
        )

        if FLAGS.dump_pdbs:
            unrelaxed_protein = protein_from_prediction(
                batch, predicted_dict)
            outfile = f'{FLAGS.outprefix}train_result_{e:02d}_{n:04d}.pdb'
            with open(outfile, 'w') as f:
                f.write(protein.to_pdb(unrelaxed_protein))
                print('made:', outfile)


        temp_train_loss.append(np.mean(loss[0]))
        temp_dscore_loss.append(np.mean(dscore_loss[0]))
        temp_lddt_ca.append(np.mean(predicted_dict['predicted_lddt']['lddt_ca']))
        temp_distogram.append(np.mean(predicted_dict['distogram']['loss']))
        temp_masked_msa.append(np.mean(predicted_dict['masked_msa']['loss']))
        temp_pred_lddt.append(np.mean(predicted_dict['predicted_lddt']['loss']))
        temp_chi_loss.append(np.mean(predicted_dict['structure_module']['chi_loss']))
        temp_fape.append(np.mean(predicted_dict['structure_module']['fape']))
        temp_sidechain_fape.append(np.mean(predicted_dict['structure_module']['sidechain_fape']))
        global_step += 1

        # accumulate
        print('grad accumulate:', global_step, grads_sum_count)
        if grads_sum_count == 0:
            grads_sum = grads
        else:
            grads_sum = jax.tree_map(lambda x, y: x+y, grads_sum, grads)
        grads_sum_count += 1

        if (grads_sum_count >= FLAGS.apply_every and # time to update
            not FLAGS.freeze_everything):
            print('grad update!', global_step, grads_sum_count)
            grads_sum = jax.tree_map(lambda x: x/grads_sum_count, grads_sum)
            grads_sum = norm_grads_per_example(grads_sum,
                                               l2_norm_clip=FLAGS.grad_norm_clip)

            #debug
            #grads_finite = jax.tree_util.tree_reduce(lambda a, b: a & jnp.all(jnp.isfinite(b)), grads_sum, initializer=True)
            #grad_norm = jnp.sqrt(sum(jnp.sum(x**2) for x in jax.tree_util.tree_leaves(grads_sum)))
            #print("grads finite:", grads_finite)
            #print("grad norm:", grad_norm)
            #debug

            updates, opt_state = gradient_transform.update(grads_sum, opt_state)
            #debug
            #updates_finite = jax.tree_util.tree_reduce(lambda a, b: a & jnp.all(jnp.isfinite(b)), updates, initializer=True)
            #update_norm = jnp.sqrt(sum(jnp.sum(x**2) for x in jax.tree_util.tree_leaves(updates)))
            #print("updates finite:", updates_finite)
            #print("update norm:", update_norm)
            #debug
            replicated_params = optax.apply_updates(replicated_params, updates)

            grads_sum, grads_sum_count = None, 0

        if (n+1) % FLAGS.print_steps == 0:
            mean_loss = round(float(np.mean(temp_train_loss)),4)
            mean_dscore_loss = round(float(np.mean(temp_dscore_loss)),4)
            lddt_ca = round(float(np.mean(temp_lddt_ca)),4)
            distogram = round(float(np.mean(temp_distogram)),4)
            masked_msa = round(float(np.mean(temp_masked_msa)),4)
            fape = round(float(np.mean(temp_fape)),4)
            sidechain_fape = round(float(np.mean(temp_sidechain_fape)),4)
            chi_loss = round(float(np.mean(temp_chi_loss)),4)
            print(f'Step: {global_step}, loss: {mean_loss}, dscore: {mean_dscore_loss}, lddt: {lddt_ca}, fape: {fape}, sc_fape: {sidechain_fape}, chi: {chi_loss}, disto: {distogram}, msa: {masked_msa}', flush=True)
            temp_train_loss = []
            temp_dscore_loss = []
            temp_lddt_ca = []
            temp_distogram = []
            temp_masked_msa = []
            temp_exper_res = []
            temp_pred_lddt = []
            temp_chi_loss = []
            temp_fape = []
            temp_sidechain_fape = []
        if (n+1) % FLAGS.save_steps == 0:
            prefix = FLAGS.outprefix
            step_fname = os.path.join(checkpoint_dir, f'{prefix}_af_mhc_global_step.npy')
            np.save(step_fname, global_step)

            param_fname = param_fname = os.path.join(checkpoint_dir, f'{prefix}_af_mhc_params_{global_step}.pkl')
            save_params = jax.tree_map(lambda x: x[0,], replicated_params)
            with open(param_fname, 'wb') as f:
                pickle.dump(save_params, f)

            state_fname = os.path.join(checkpoint_dir, f'{prefix}_af_mhc_state_{global_step}.pkl')
            with open(state_fname, 'wb') as f:
                pickle.dump(opt_state, f)

            config_fname = os.path.join(checkpoint_dir, f'{prefix}_af_mhc_state_{global_step}.pkl')
            with open(config_fname, 'wb') as f:
                pickle.dump(model_config, f)

        if (n+1)%FLAGS.valid_steps==0:
            compute_valid_stats(valid_loader, replicated_params, jax_key)
        sys.stdout.flush()

    compute_valid_stats(valid_loader, replicated_params, jax_key)
