#!/bin/bash
#SBATCH --job-name=af_driver
#SBATCH --output=logs/driver_%j.out
#SBATCH --error=logs/driver_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00

#Copyright (c) 2026 The Children's Hospital of Philadelphia and Stanford University
#Licensed for academic and non-commercial use only. Commercial use requires a separate license.
#See LICENSE file for details.

set -euo pipefail
mkdir logs
# --------------------------------------------------------------------------
# Edit these for your run (relative to the directory you launch this from)
# --------------------------------------------------------------------------
INPUT_SEQ_DIR=input_seq                          # initialize.py --input-seq-dir
TEMPLATE_PDB_DIR=template_pdbs                   # initialize.py --template-pdb-dir
TARGETS_ROOT=outfiles                              # initialize.py --output-root
PARAMS_FILE=affthla3db.pkl
MODEL_NAME=model_2_ptm                           # match run_prediction.py --model_name if overridden
WORKER=predict_structure.sh                            # per-target sbatch worker (used only with --parallel)

# Extra flags passed to initialize.py, e.g.:
# GENERATE_ARGS="--top-n 6 --min-peptide-mismatches 2"
GENERATE_ARGS=""

# Extra flags passed to run_prediction.py, e.g.:
# EXTRA_ARGS="--num_recycle 3 --resample_msa"
EXTRA_ARGS=""

# --------------------------------------------------------------------------
# Environment (edit for your cluster: module load / conda / venv)
# --------------------------------------------------------------------------
# Parse our own args FIRST, before touching conda, and clear $@ afterward.
PARALLEL=0
for arg in "$@"; do
    case "$arg" in
        --parallel) PARALLEL=1 ;;
    esac
done
set --   # clear positional args so nothing downstream (conda) inherits them

# Now activate conda, with set -u relaxed (conda scripts trip -u)
set +u
conda activate afft_hla3db
set -u

# --------------------------------------------------------------------------
# 1) Generate all per-target input folders, here on this node.
# --------------------------------------------------------------------------
echo "Generating per-target inputs into ${TARGETS_ROOT}/ ..."
python initialize.py \
    --input-seq-dir "${INPUT_SEQ_DIR}" \
    --template-pdb-dir "${TEMPLATE_PDB_DIR}" \
    --output-root "${TARGETS_ROOT}" \
    ${GENERATE_ARGS}

TARGETS_ROOT_ABS=$(realpath "${TARGETS_ROOT}")
PARAMS_FILE_ABS=$(realpath "${PARAMS_FILE}")

mapfile -t TARGET_IDS < <(find "${TARGETS_ROOT_ABS}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
if [[ "${#TARGET_IDS[@]}" -eq 0 ]]; then
    echo "No targets found under ${TARGETS_ROOT_ABS} after generation." >&2
    exit 1
fi

# --------------------------------------------------------------------------
# 2) Dispatch prediction.
# --------------------------------------------------------------------------
if [[ "${PARALLEL}" -eq 1 ]]; then
    echo "Submitting ${#TARGET_IDS[@]} independent sbatch jobs (one per target)..."
    for target_id in "${TARGET_IDS[@]}"; do
        sbatch "${WORKER}" "${target_id}" "${TARGETS_ROOT_ABS}" "${PARAMS_FILE_ABS}" "${MODEL_NAME}" "${EXTRA_ARGS}"
    done
    echo "All jobs submitted."
else
    echo "Running ${#TARGET_IDS[@]} targets sequentially on $(hostname)..."
    for target_id in "${TARGET_IDS[@]}"; do
        target_tsv="${TARGETS_ROOT_ABS}/${target_id}/inputs/target.tsv"
        outdir="${TARGETS_ROOT_ABS}/${target_id}/outputs"
        mkdir -p "${outdir}"
        echo "=== ${target_id} ==="
        python run_prediction.py \
            --targets "${target_tsv}" \
            --params_file "${PARAMS_FILE_ABS}" \
            --outfile_prefix "${target_id}" \
            --output_dir "${outdir}" \
            --model_name "${MODEL_NAME}" \
            --verbose \
            ${EXTRA_ARGS}
    done
    echo "All targets done."
fi