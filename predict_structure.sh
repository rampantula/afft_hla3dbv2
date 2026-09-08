#!/bin/bash
#SBATCH --job-name=afft_hla3db
#SBATCH --time=0:30:00
#SBATCH -p gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --output=logs/predict_%j.out
#SBATCH --error=logs/predict_%j.err

#Copyright (c) 2026 The Children's Hospital of Philadelphia and Stanford University
#Licensed for academic and non-commercial use only. Commercial use requires a separate license.
#See LICENSE file for details.


# Per-target prediction worker. Not run by hand -- the driver
# (fold.sh --parallel) submits one of these per target:
#
#   sbatch predict_structure.sh <target_id> <targets_root_abs> <params_file_abs> <model_name> <extra_args>
#
# Writes outputs into <targets_root_abs>/<target_id>/outputs/

set -euo pipefail
# --------------------------------------------------------------------------
# Environment (edit for your cluster: module load / conda / venv)
# --------------------------------------------------------------------------

# Parallel XLA compilation across the CPUs SLURM gave us (fallback to 12)
N_THREADS=${SLURM_CPUS_PER_TASK:-12}
export XLA_FLAGS="--xla_gpu_force_compilation_parallelism=${N_THREADS} --xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false"

# Quiet the noisy startup warnings
export TF_CPP_MIN_LOG_LEVEL=2
export TF_ENABLE_ONEDNN_OPTS=0

echo "XLA_FLAGS=$XLA_FLAGS | CPUs=$(nproc)"

TARGET_ID="$1"
TARGETS_ROOT_ABS="$2"
PARAMS_FILE_ABS="$3"
MODEL_NAME="$4"
EXTRA_ARGS="${5:-}"

echo "[stage] activating conda ..."
set +ux
conda activate afft_hla3db
set -ux

TARGET_TSV="${TARGETS_ROOT_ABS}/${TARGET_ID}/inputs/target.tsv"
if [[ ! -f "${TARGET_TSV}" ]]; then
    echo "Missing ${TARGET_TSV}" >&2
    exit 1
fi

OUTDIR="${TARGETS_ROOT_ABS}/${TARGET_ID}/outputs"
mkdir -p "${OUTDIR}"

echo "target=${TARGET_ID} on $(hostname)"

python run_prediction.py \
    --targets "${TARGET_TSV}" \
    --params_file "${PARAMS_FILE_ABS}" \
    --outfile_prefix "${TARGET_ID}" \
    --output_dir "${OUTDIR}" \
    --model_name "${MODEL_NAME}" \
    --verbose \
    ${EXTRA_ARGS}