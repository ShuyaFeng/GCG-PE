"""GCG-based extraction adapted to PII targets, via nanoGCG.

For each (individual, field) we optimize a prompt so the model reproduces the
labeled target string. We log:
  * one Attempt (method='gcg') with the final per-field hit (single source of truth)
  * convergence rows (hit rate by iteration checkpoint)
  * the optimized prompt string (for transfer + prompt-property analysis)
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import nanogcg
import torch
from nanogcg import GCGConfig

from .data_gen import target_string
from .genutils import greedy_generate
from .utils import Attempt, ResultsStore, field_hit

PLACEHOLDER = "{optim_str}"


def _gcg_config(g: Dict[str, Any], seed: int) -> GCGConfig:
    return GCGConfig(
        num_steps=g["num_steps"],
        search_width=g["search_width"],
        topk=g["topk"],
        optim_str_init=" ".join(["!"] * g["optim_tokens"]),
        early_stop=bool(g.get("early_stop", True)),
        seed=seed,
        verbosity="WARNING",
    )


def _best_string_upto(result, step: int) -> Optional[str]:
    """Best optimized string among the first `step` iterations."""
    strings = getattr(result, "strings", None)
    losses = getattr(result, "losses", None)
    if not strings or not losses:
        return getattr(result, "best_string", None)
    n = min(step, len(strings), len(losses))
    if n <= 0:
        return getattr(result, "best_string", None)
    best_i = min(range(n), key=lambda i: losses[i])
    return strings[best_i]


def run_gcg(model, tok, model_name: str, seed: int,
            individuals: List[Dict[str, Any]], fields: List[str],
            cfg: Dict[str, Any], store: ResultsStore,
            device: str = "cuda") -> Dict[str, Dict[str, str]]:
    """Returns {f"{individual_id}:{field}": optimized_prompt} for transfer."""
    g = cfg["extract"]["gcg"]
    max_new = cfg["extract"]["max_new_tokens"]
    checkpoints = sorted(g["checkpoints"])
    optimized_prompts: Dict[str, str] = {}

    for ind in individuals:
        for field in fields:
            value = ind[field]
            target = target_string(field, ind)
            t0 = time.time()
            result = nanogcg.run(model, tok, PLACEHOLDER, target,
                                 _gcg_config(g, seed))
            runtime = time.time() - t0

            # Convergence: hit rate at each checkpoint.
            final_hit, final_prompt, final_out = False, "", ""
            for c in checkpoints:
                s = _best_string_upto(result, c)
                prompt = s if s is not None else ""
                out = greedy_generate(model, tok, prompt, max_new, device)
                hit = field_hit(out, value, field)
                store.add_gcg_convergence({
                    "model": model_name, "seed": seed,
                    "individual_id": ind["id"], "field": field,
                    "frequency": ind["frequency"], "iteration": c, "hit": hit,
                })
                final_hit, final_prompt, final_out = hit, prompt, out

            key = f"{ind['id']}:{field}"
            optimized_prompts[key] = final_prompt
            store.add_attempt(Attempt(
                model=model_name, seed=seed, individual_id=ind["id"],
                field=field, frequency=ind["frequency"], method="gcg",
                hit=final_hit, best_prompt=final_prompt, output=final_out,
                runtime_s=runtime,
            ))
            store.add_prompt({
                "model": model_name, "seed": seed, "method": "gcg",
                "field": field, "individual_id": ind["id"],
                "prompt": final_prompt, "success": final_hit, "kind": "gcg",
            })
    return optimized_prompts


def run_transfer(target_model, target_tok, target_name: str,
                 source_name: str, seed: int,
                 individuals: List[Dict[str, Any]], fields: List[str],
                 optimized_prompts: Dict[str, str], cfg: Dict[str, Any],
                 store: ResultsStore, device: str = "cuda") -> None:
    """Apply prompts optimized on `source` to a frozen `target` model."""
    max_new = cfg["extract"]["max_new_tokens"]
    by_ind: Dict[int, Dict[str, bool]] = {}
    for ind in individuals:
        by_ind.setdefault(ind["id"], {})
        for field in fields:
            prompt = optimized_prompts.get(f"{ind['id']}:{field}")
            if prompt is None:
                continue
            out = greedy_generate(target_model, target_tok, prompt, max_new,
                                  device)
            hit = field_hit(out, ind[field], field)
            by_ind[ind["id"]][field] = hit
            store.add_transfer({
                "source": source_name, "target": target_name, "seed": seed,
                "individual_id": ind["id"], "field": field,
                "frequency": ind["frequency"], "hit": hit, "level": "field",
            })
