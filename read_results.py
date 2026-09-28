#!/usr/bin/env python3
"""Summarize the v3 results and say what each number means for the paper.

    python3 read_results.py [results_dir]   # default: runs/v3, else results_v3

Reads whatever is present and skips what is not, so it is useful even if only
one job finished. For every quantity it prints the measured value, the value the
paper currently prints, and the edit that follows. Nothing is written.
"""
from __future__ import annotations

import json
import os
import sys

if len(sys.argv) > 1:
    DIR = sys.argv[1]
else:
    # On the cluster the results sit in runs/v3; on a laptop they were rsynced
    # into results_v3. Take whichever exists so the command is the same in both.
    DIR = next((d for d in ("runs/v3", "results_v3") if os.path.isdir(d)), "runs/v3")

# What the paper currently asserts, so a mismatch is visible rather than implied.
PAPER = {
    "blind_pct": 2.1,       # Sec 4.3: target-blind bound at the GCG budget, m_S=184
    "design_bits": 29.8,    # Cor. 1.3: min-entropy needed at that budget
    "h_inf_ssn": 28.73,
    "h_inf_email": 13.4,
    "m_s_rigorous": 760,
    "m_s_typical": 184,
    "sep_looser": 9,        # "at least ninefold"
    "sep_tighter": 36,      # "thirty-sixfold at the tighter multiplicity"
    "b_base": -4.7,
    "did": 8.7,
    "tau_mod": 28.7,
    "tau_rec": 4.0,
    "beta_eff": 3.0,
    "quantiles": (6.6, 9.5, 17.1),
}


def load(name):
    p = os.path.join(DIR, name)
    if not os.path.exists(p):
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except Exception as exc:
        print(f"  ! {name} is present but unreadable: {exc}")
        return None


def head(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def verdict(text):
    print()
    for line in text.strip().split("\n"):
        print("  -> " + line.strip())


def near(a, b, tol):
    try:
        return abs(float(a) - float(b)) <= tol
    except Exception:
        return False


# --------------------------------------------------------------------------- #
found_any = False

# ---------- 1. theory constants -------------------------------------------- #
t = load("theory.json")
if t:
    found_any = True
    head("1. THEORY CONSTANTS  (confirms Sec. 4.3 and Cor. 1.3; no experiment)")
    L = t["multiplicity"]["decode_len_L"]
    Q = t["gcg_query_budget"]
    print(f"  decode length L        : {L}")
    print(f"  GCG query budget Q     : {Q}   (= steps x search_width)")
    print(f"  m_S rigorous / typical : {t['multiplicity']['m_s_rigorous']} / "
          f"{t['multiplicity']['m_s_typical']}")
    for e in t["entropy"]:
        flag = "" if e.get("certified") else "   [NOT certified: do not put in a theorem]"
        print(f"  H_inf({e['field']:<5})          : {e['h_inf_bits']:.3f} bits{flag}")

    row = next((b for b in t["bounds"]
                if b["field"] == "ssn" and b["m_S"] == "m_S=184"), None)
    if row:
        blind = 100 * row["blind_Q_gcg"]
        bits = row["design_bits_gcg"]
        print()
        print(f"  target-blind bound at Q (ssn, m_S=184) : {blind:.2f}%"
              f"      paper prints {PAPER['blind_pct']}%")
        print(f"  min-entropy needed at Q                : {bits:.1f} bits"
              f"   paper prints {PAPER['design_bits']} bits")
        print(f"  k_vac                                  : {row['k_vac']}"
              f"                paper prints 2")

        rows760 = next((b for b in t["bounds"]
                        if b["field"] == "ssn" and b["m_S"] == "m_S=760"), None)
        a20 = t.get("measured_alpha_20", 0.7733)
        sep_t = a20 / row["blind_Q_gcg"] if row["blind_Q_gcg"] else float("inf")
        sep_l = (a20 / rows760["blind_Q_gcg"]) if rows760 and rows760["blind_Q_gcg"] else float("nan")
        print(f"  separation vs measured {100*a20:.1f}%          : "
              f"{sep_l:.0f}x (m_S=760, looser) and {sep_t:.0f}x (m_S=184, tighter)")
        print(f"                                           paper prints "
              f"{PAPER['sep_looser']}x and {PAPER['sep_tighter']}x")

        ok = (near(blind, PAPER["blind_pct"], 0.15)
              and near(bits, PAPER["design_bits"], 0.2)
              and near(sep_l, PAPER["sep_looser"], 1.5)
              and near(sep_t, PAPER["sep_tighter"], 3))
        if ok:
            verdict("MATCHES the paper. Nothing to change in Sec. 4.3 or Cor. 1.3.")
        else:
            verdict(f"""
                MISMATCH. Your run used L={L} and Q={Q}, which is not what the paper
                assumed. Three places must be updated together, or Sec. 4.3 forks:
                  (a) Sec. 4.3 bound: {PAPER['blind_pct']}% -> {blind:.1f}%
                  (b) Sec. 4.3 separation: {PAPER['sep_looser']}x/{PAPER['sep_tighter']}x -> {sep_l:.0f}x/{sep_t:.0f}x
                  (c) Cor. 1.3 commentary: {PAPER['design_bits']} bits -> {bits:.1f} bits
                Send me these numbers and I will apply all three.
                """)
else:
    print("  (no theory.json)")

# ---------- 2. memorization verification ----------------------------------- #
e5 = load("e5_bits.json")
if e5:
    found_any = True
    head("2. MEMORIZATION VERIFICATION  (decides the framing of Sec. 6.1)")
    summary = e5.get("summary", {})
    if not summary:
        print("  e5_bits.json has no summary block; the run may have died early.")
    rates = []
    for field, v in summary.items():
        tp = v.get("trueprefix_greedy_hit_rate_ft")
        rates.append(tp)
        print(f"  {field:<6} n={v.get('n','?'):<4} "
              f"true-prefix greedy hit {100*tp:5.1f}%   "
              f"NLL true-prefix {v.get('median_nll_bits_trueprefix_ft', float('nan')):6.1f} bits   "
              f"NLL neutral {v.get('median_nll_bits_neutral_ft', float('nan')):7.1f} bits")
    rates = [r for r in rates if r is not None]
    worst = max(rates) if rates else 0.0

    if worst >= 0.5:
        verdict(f"""
            The premise HOLDS: the true training prefix recovers up to
            {100*worst:.0f}% of trained targets by greedy decoding alone.
            This is good news and it STRENGTHENS the paper.
            Edit: Sec. 6.1 currently says "we report no per-record verification of
            verbatim retention". Replace that with the measured rate, and drop the
            hedge. The NLL gap between the two columns is also a reportable
            bits-supplied decomposition.
            """)
    elif worst >= 0.1:
        verdict(f"""
            PARTIAL: {100*worst:.0f}% recovery from the true prefix. Report it as
            measured, keep the hedge, and say that retention is verbatim for a
            minority of records. No reframing needed.
            """)
    else:
        verdict(f"""
            The premise does NOT hold as stated: only {100*worst:.0f}% of trained
            targets are recovered from their own training prefix.
            That is itself a finding, and the paper's current wording already
            survives it, so nothing is broken. Worth doing: state this 0 as a
            result with one verbatim prompt/generation example, and let the framing
            be "the audit reports forcing where little is verbatim-extractable".
            """)

    dk = e5.get("delta_k")
    if dk:
        vals = [r.get("bits_per_free_token") for r in dk
                if isinstance(r, dict) and r.get("bits_per_free_token") is not None]
        if vals:
            vals.sort()
            med = vals[len(vals) // 2]
            print()
            print(f"  measured bits supplied per free token (median over "
                  f"{len(vals)} attempts): {med:.2f}")
            verdict(f"""
                This is a DIRECT measurement of what Sec. 6.4 currently infers
                from k_50, where it prints beta_eff = {PAPER['beta_eff']} bits per
                free token. Measured {med:.2f}. If these disagree, prefer the
                measurement and I will rewrite that sentence.
                """)
else:
    print("  (no e5_bits.json)")

# ---------- 3. diagnostics ------------------------------------------------- #
d = load("diag.json")
if d:
    found_any = True
    head("3. COPY DIAGNOSTIC  (the largest validity threat to the paper)")
    copy = d.get("copy") or {}
    if copy.get("error"):
        print(f"  NOT RUN: {copy['error']}")
        verdict("""
            The log lacks prompt_text, so this could not be computed. Fall back to
            the two-minute eyeball check in scripts/v3_diagnostics.sbatch: print 20
            successful optimized prompts and look for the target's digits. Either
            way the paper stands as written, because Limitations already concedes
            the mechanism is undiagnosed.
            """)
    elif copy:
        overall = copy.get("overall", {})
        if overall:
            print(f"  overall: n_hits={overall.get('n_hits')}  "
                  f"full_copy_rate={overall.get('full_copy_rate', float('nan')):.3f}")
        for r in copy.get("rows", []):
            print(f"  {str(r.get('probe','?')):<16} k={str(r.get('capacity_k','?')):<4} "
                  f"hits={r.get('n_hits','?'):<5} "
                  f"full={r.get('full_copy_rate', float('nan')):.2f} "
                  f"partial={r.get('partial_copy_rate', float('nan')):.2f} "
                  f"NON-COPY={r.get('non_copy_rate', float('nan')):.2f}")
        nc = [r.get("non_copy_rate") for r in copy.get("rows", [])
              if r.get("non_copy_rate") is not None]
        if nc:
            lo = min(nc)
            if lo >= 0.7:
                verdict(f"""
                    COPYING IS NOT THE MECHANISM: at least {100*lo:.0f}% of hits
                    come from prompts that do not contain their own target.
                    Edit: add one sentence to Sec. 6.1. This single sentence blocks
                    the strongest attack a reviewer has on this paper.
                    """)
            elif lo >= 0.3:
                verdict(f"""
                    MIXED: non-copy share as low as {100*lo:.0f}% in some cell.
                    Report the non-copy rate as the headline forcing rate and say
                    copying accounts for the rest. A real result, not a problem.
                    """)
            else:
                verdict(f"""
                    COPYING DOMINATES (non-copy share {100*lo:.0f}%).
                    The mechanism story must change, and the replacement is
                    SHARPER: forcing here is copy-through-prompt, and the
                    prescription becomes a constraint on probe length relative to
                    target length, which connects directly to adversarial
                    compression. Tell me and I will rewrite Sec. 4.2's mechanism
                    paragraph, Sec. 6.1 and the abstract. Budget about two hours.
                    """)

    # free extras that need no new run
    for key, title, note in [
        ("feasibility", "OPERATING-POINT FEASIBILITY",
         "The paper predicts this set is EMPTY: wherever the floor is low the "
         "trained arm is zero too. If it is empty, that is a publishable negative "
         "result and it is already argued in the Discussion."),
        ("twobytwo", "2x2 PLACEBO AND DiD",
         f"Cross-check the paper's numbers: b_base={PAPER['b_base']}, "
         f"DiD={PAPER['did']}, tau_mod={PAPER['tau_mod']}, tau_rec={PAPER['tau_rec']}."),
        ("field", "PER-FIELD SPLIT (SSN vs email)",
         "This is what finalizes Remark 'a linear bits-per-token model is "
         "inconsistent with the curve'. If the SSN-only arm still shows a wide "
         "k_75/k_25 spread, the remark can drop its pooling caveat."),
        ("seed", "SEED vs TARGET VARIANCE",
         "Mostly 0/3 and 3/3 means elicitability is a property of the target and "
         "k_min is meaningful. Many 1/3 and 2/3 means it is optimizer noise, and "
         "any k_min-based claim must go."),
        ("acr", "ACR ON CONTROLS",
         "The share of never-trained targets reachable by a prompt shorter than "
         "the target. A non-trivial share falsifies the field's standard forcing "
         "guard and could become a main-table result."),
        ("scoring", "SCORING GUARDS",
         "The random-record match rate the paper calls 'near zero' should have a "
         "number here."),
    ]:
        block = d.get(key)
        if block:
            head(f"EXTRA: {title}")
            if isinstance(block, dict) and block.get("error"):
                print(f"  not available: {block['error']}")
            else:
                s = json.dumps(block, indent=2, default=str)
                print(s[:1600] + (" ...[truncated]" if len(s) > 1600 else ""))
            verdict(note)
else:
    print("  (no diag.json)")

if not found_any:
    print()
    print("Nothing found in", os.path.abspath(DIR))
    print("Pass the directory as an argument, e.g. python3 read_results.py runs/v3")
else:
    head("WHAT TO SEND ME")
    print("  Paste this whole output, or just say which of the three verdicts you got.")
    print("  I will apply the corresponding edits to satml2027_paper_v2.tex.")
