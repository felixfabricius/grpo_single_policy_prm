#!/bin/bash
#SBATCH --job-name=grpo1
#SBATCH --clusters=srf_gpu_01
#SBATCH --partition=standard-gpu
#SBATCH --gres=gpu:Ada_RTX_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --time=24:00:00
#SBATCH --mail-user=felix.fabricius@lmh.ox.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL

cd "$SLURM_SUBMIT_DIR"

# ---- User knobs ----
RUN_ID="${RUN_ID:-run_$(date -u +%Y-%m-%d_%H-%M-%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-runs}"
CONFIG_PATH="${CONFIG_PATH:-configs/train.yaml}"

# Create run dir early (for logs)
mkdir -p "${OUTPUT_DIR}/${RUN_ID}"

# Fast local caches (fallback to $SCRATCH if SLURM_TMPDIR missing)
if [[ -n "${SLURM_TMPDIR}" ]]; then
  export HF_HOME="${SLURM_TMPDIR}/hf"
  export TRANSFORMERS_CACHE="${SLURM_TMPDIR}/hf/transformers"
  export HF_DATASETS_CACHE="${SLURM_TMPDIR}/hf/datasets"
else
  export HF_HOME="${SCRATCH:-$HOME}/.cache/hf"
  export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
  export HF_DATASETS_CACHE="${HF_HOME}/datasets"
fi
mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${HF_DATASETS_CACHE}"

# ---- Activate environment ----
# Option A: Conda
# source ~/.bashrc
# conda activate my-grpo-env

# Option B: venv
# source /path/to/venv/bin/activate

# ---- Log to runs/${RUN_ID}/slurm-%j.out ----
LOG_PATH="${OUTPUT_DIR}/${RUN_ID}/slurm-${SLURM_JOB_ID}.out"
echo "[INFO] Logging to ${LOG_PATH}"

set -xeuo pipefail

srun python -u grpo_single_policy_prm/scripts/train.py \
  --config "${CONFIG_PATH}" \
  --run-id "${RUN_ID}" \
  --output-dir "${OUTPUT_DIR}" \
  --max-steps 50 \
  |& tee -a "${LOG_PATH}"
