#!/bin/bash
# SLURM submit script for UAB Cheaha (single A100 80GB).
#
#   Submit:   sbatch scripts/cheaha_run.sbatch configs/full.yaml
#   Smoke:    sbatch scripts/cheaha_run.sbatch configs/smoke.yaml
#
# The full run is ~1000 GPU-hours, which exceeds a single job's walltime. The
# pipeline is RESUMABLE: just resubmit this script and it continues where it
# left off (finished models/targets are skipped, no duplicate logging). See the
# self-resubmit block at the bottom to chain automatically.

#SBATCH --job-name=gcg-pe
#SBATCH --partition=amperenodes          # A100 80GB partition on Cheaha
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00                   # bump toward the partition max if allowed
#SBATCH --output=logs/gcgpe_%j.out
#SBATCH --error=logs/gcgpe_%j.out
## #SBATCH --account=YOUR_ALLOCATION       # uncomment if Cheaha requires an account

set -euo pipefail
mkdir -p logs
cd "$SLURM_SUBMIT_DIR"

CONFIG="${1:-configs/full.yaml}"

# CUDA: the pip torch wheel bundles its own CUDA runtime, so a compatible driver
# (present on GPU nodes) is enough. Load a module only if you build against system CUDA.
# module load CUDA/12.2.0 || true

source .venv/bin/activate
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"   # cache datasets/models in-project (home has space)
export TOKENIZERS_PARALLELISM=false

echo "[$(date)] host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-?} config=${CONFIG}"
nvidia-smi || true

python run.py --config "${CONFIG}"

# ----------------------------------------------------------------------------
# Optional auto-chaining: if the full run is not finished, resubmit to continue.
# Heuristic: stop chaining once the tables exist. Comment out if you prefer
# to resubmit manually.
# ----------------------------------------------------------------------------
EXP_DIR=$(python -c "import yaml,sys;print(yaml.safe_load(open('${CONFIG}'))['output_dir'])")
if [ ! -f "${EXP_DIR}/tables/tab_main_results.tex" ]; then
  echo "[$(date)] run incomplete -> resubmitting to continue (resumable)"
  sbatch "$0" "${CONFIG}"
else
  echo "[$(date)] tables present -> done, not resubmitting"
fi
