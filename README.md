# Reproducible Privacy-Auditing Pipeline

Regenerates **every table** in the paper from a **single per-attempt log**, so
record-level (Table 1) and per-field (Tables 2/3) metrics are guaranteed
consistent by construction. This fixes the original Table 1 ↔ Table 2 mismatch
(two different `exact_match` definitions) at the source.

## What it does
1. **data** – synthesize PII corpus with full ground truth (`src/data_gen.py`).
2. **train** – fine-tune each model (full FT; LoRA for Llama-2-7B) (`src/train.py`).
3. **extract** – 4 fixed-prompt baselines + GCG (nanoGCG) per `(individual, field)`;
   logs one per-field hit per method (`src/extract_baselines.py`, `src/extract_gcg.py`).
4. **transfer** – apply source-optimized prompts to target models.
5. **validate** – natural-memorization check on GPT-2-XL (`data/memorized_sequences.json`).
6. **features** – 24 linguistic features + record-level success for the primary model.
7. **tables** – emit all `*.tex` tables to `runs/<exp>/tables/`.

## The single source of truth
`runs/<exp>/attempts.jsonl` — one row per `(model, seed, individual, field, method)`
with a per-field `hit`. Everything else is a *view*:
- **record-level** (Table 1): `name ∧ ssn ∧ email` all hit, per individual.
- **per-field** (Tables 2/3/8): mean over field attempts.
- `baseline` = OR over the four fixed-prompt methods; `optimized` = GCG.

Change `extract.record_fields` in the config to redefine record-level success in
**one** place; all tables follow.

## Setup (once)
```bash
cd pipeline
bash setup.sh                     # venv + torch + deps + spaCy model
source .venv/bin/activate
huggingface-cli login             # only needed for Llama-2-7B access
```

## Run (zero manual data steps)
```bash
# 1) Smoke test first (minutes, one small model, offline) — verifies everything:
python run.py --config configs/smoke.yaml

# 2) Full run (single A100/H100 80GB; ~1000 GPU-hours, resumable):
python run.py --config configs/full.yaml
```
The full run **downloads the real public corpus automatically** (Wikipedia /
PG-19 / arXiv via HuggingFace `datasets`, streamed + cached to
`runs/full/data/cache/`). Any source that fails to download is skipped; if the
box is fully offline it falls back to synthetic filler with a warning.

Outputs land in `runs/<exp>/`: `attempts.jsonl` (source of truth),
`tables/*.tex`, trained models. Paste `tables/*.tex` into `latex/` — labels
already match the paper.

```bash
# Re-run a single stage, e.g. rebuild tables after a formatting tweak:
python run.py --config configs/full.yaml --stages tables
```

## Dataset details
- **Public corpus**: configured under `data.public_sources` in `configs/full.yaml`.
  Edit that list to change sources; defaults need no auth. First run downloads,
  later runs reuse the cache.
- **Natural-memorization validation** (`data/memorized_sequences.json`): ships ~25
  well-known public strings (URLs, license boilerplate, code, famous text) that
  GPT-2-XL plausibly memorized, so Table 5 runs out of the box. Add the canonical
  Carlini et al. memorized set here for the strongest validation.

## Notes
- GCG is the cost driver. `extract.gcg.num_steps` and `search_width` trade quality
  for compute; `checkpoints` controls the convergence-table granularity.
- bf16 is on by default (A100/H100). Set `train.bf16: false, fp16: true` on older GPUs.
