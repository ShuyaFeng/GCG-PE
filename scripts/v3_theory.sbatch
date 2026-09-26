#!/bin/bash
# STEP 0 of the v3 minimum set: the theory constants for Section 4.
#
# NO GPU, NO EXPERIMENT. This is exact enumeration of the Faker generator and the
# GPT-2 vocabulary, so it is pure computation and takes seconds. It exists as a
# batch job only so the numbers land in a log you can cite; running it on the
# login node is equally fine:
#
#   python -m src.theory --decode-len 48 --search-width 256
#
# Submit (pass YOUR actual run constants, not the defaults):
#   sbatch scripts/v3_theory.sbatch <decode_len_L> <search_width> [gcg_steps]
#
# Example, if your sweep decoded 48 tokens with search width 256 over 200 steps:
#   sbatch scripts/v3_theory.sbatch 48 256 200
#
# WHERE TO FIND THE TWO CONSTANTS (they are NOT in the v2 draft anywhere):
#   decode_len_L  = extract.max_new_tokens in the config that produced the sweep
#   search_width  = extract.gcg.search_width in that same config
#
# WHAT TO DO WITH THE OUTPUT:
#   Paste `blind Q=gcg` and `need bits` for the ssn/m_S=184 row into
#   satml2027_paper_v2.tex Section 4.3. The file has a
#   "% CONFIRM BEFORE SUBMISSION" comment marking the exact spot.

#SBATCH --job-name=v3-theory
#SBATCH --partition=express
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/v3_theory_%j.out
#SBATCH --error=logs/v3_theory_%j.out
## #SBATCH --account=YOUR_ALLOCATION

set -uo pipefail
mkdir -p logs runs/v3
cd "$SLURM_SUBMIT_DIR"

DECODE_LEN="${1:?usage: sbatch scripts/v3_theory.sbatch <decode_len_L> <search_width> [gcg_steps]}"
SEARCH_WIDTH="${2:?need the search width your sweep actually used}"
GCG_STEPS="${3:-200}"

source .venv/bin/activate
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
export TOKENIZERS_PARALLELISM=false

echo "[$(date)] host=$(hostname)"
echo "[$(date)] L=${DECODE_LEN} search_width=${SEARCH_WIDTH} steps=${GCG_STEPS}"
echo

python -m src.theory \
    --decode-len "${DECODE_LEN}" \
    --search-width "${SEARCH_WIDTH}" \
    --gcg-steps "${GCG_STEPS}" \
    --json runs/v3/theory.json

echo
echo "[$(date)] done -> runs/v3/theory.json"
echo "Paste the ssn / m_S=184 row's 'blind Q=gcg' and 'need bits' into Section 4.3."
