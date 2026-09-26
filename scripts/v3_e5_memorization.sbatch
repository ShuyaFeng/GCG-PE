#!/bin/bash
# STEP 2 of the v3 minimum set: memorization verification (experiment E5).
#
# WHY THIS ONE IS WORTH A GPU JOB
# The paper's load-bearing premise is "the content IS memorized". Right now the
# only evidence for it is a soft prompt that recovers 100% of trained targets --
# and also 100% of controls, so by the paper's own argument it carries zero
# attribution information. Meanwhile every fixed prompt, including a true-prefix
# completion, recovers 0%. That combination is not currently explainable, and
# this job resolves it.
#
# It runs NO optimization: only teacher-forced target NLL and greedy completion
# from the true training-template prefix, on checkpoints you already have.
# Roughly 400 forward passes for 100 targets on two checkpoints.
#
# Submit:
#   sbatch scripts/v3_e5_memorization.sbatch <finetuned_ckpt> <individuals.json> [controls.json]
#
# Example:
#   sbatch scripts/v3_e5_memorization.sbatch \
#       runs/full_fast/models/gpt2-seed0 runs/full_fast/data/individuals.json
#
# If you omit controls.json the job generates one first (matched, entropy known
# exactly by construction, and drawn from identifier ranges that cannot collide
# with real SSNs).
#
# HOW TO READ THE RESULT
#   true-prefix greedy hit rate HIGH  -> the premise holds; cite it as measured
#                                        and drop the soft-prompt argument.
#   true-prefix greedy hit rate ~0    -> that is itself the finding. Reframe as
#                                        "the audit reports forcing where little
#                                        is verbatim-extractable", and delete the
#                                        "content IS memorized" paragraph in 6.1.
# Either way the circularity goes away.

#SBATCH --job-name=v3-e5
#SBATCH --partition=amperenodes
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/v3_e5_%j.out
#SBATCH --error=logs/v3_e5_%j.out
## #SBATCH --account=YOUR_ALLOCATION

set -uo pipefail
mkdir -p logs runs/v3
cd "$SLURM_SUBMIT_DIR"

CKPT="${1:?usage: sbatch scripts/v3_e5_memorization.sbatch <finetuned_ckpt> <individuals.json> [controls.json]}"
INDIV="${2:?need the individuals.json that was used to build the fine-tuning corpus}"
CONTROLS="${3:-runs/v3/controls.json}"
BASE_MODEL="${BASE_MODEL:-gpt2}"

for f in "${CKPT}" "${INDIV}"; do
  if [ ! -e "$f" ]; then echo "ERROR: no such path: $f"; exit 2; fi
done

source .venv/bin/activate
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
export TOKENIZERS_PARALLELISM=false

echo "[$(date)] host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-?}"
echo "[$(date)] ckpt=${CKPT} base=${BASE_MODEL}"
nvidia-smi || true

# ---- controls: generate if not supplied ----
if [ ! -f "${CONTROLS}" ]; then
  echo
  echo "================ generating matched controls ================"
  python -m src.controls -n 200 --seed 99991 \
      --individuals "${INDIV}" \
      --tokenizer "${BASE_MODEL}" --ref-model distilgpt2 \
      --fields ssn email \
      --out "${CONTROLS}"
fi

echo
echo "================ E5: memorization verification ================"
# float32 on purpose: NLL in float16 is too noisy to compare arms.
python experiments/e5_nll_decomposition.py \
    --model "${CKPT}" \
    --base-model "${BASE_MODEL}" \
    --individuals "${INDIV}" \
    --controls "${CONTROLS}" \
    --fields ssn email \
    --dtype float32 \
    --out runs/v3/e5_bits.json

echo
echo "[$(date)] done -> runs/v3/e5_bits.json"
echo "Look at the true-prefix greedy hit rate for the D arm first; it decides"
echo "whether Section 6.1's memorization paragraph stands or gets rewritten."
