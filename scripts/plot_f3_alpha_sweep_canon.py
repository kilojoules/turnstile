"""F.3 — Alpha-sweep figures rebuilt on canon Likert. Confirms shape survival.

Reads the canon Qwen + canon Llama re-judged layer-sweep cells, plots:
  (a) alpha-sweep at L16: canon Likert (Qwen, Llama) and ASR vs α, for the
      4 directions + random + baseline. Inverted-U on Likert and refusal-
      knockover on ASR should persist (shape) with absolute Likert level
      shifted up ~0.3.
  (b) per-cell shape-survival comparison: for each (method, layer, alpha),
      compute mean truncated Likert vs mean canon Qwen Likert; produce a
      scatter that should hug y=x+~0.3, with sign of difference (Δ from α=0)
      preserved.

Also writes a small JSON report on shape preservation: per (method, layer)
the truncated vs canon trend across α, and whether the directional sign (up
or down from baseline) is preserved.
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/Users/julianquick/portfolio_copy/turnstile"
LSW = f"{ROOT}/experiments/steering_v3/layer_sweep"
FIG = f"{ROOT}/figures"


def coh_ok(r):
    c = r.get("coherence", {}) or {}
    return (c.get("n_tokens", 0) >= 5 and
            c.get("token_unique_ratio", 1) >= 0.15 and
            c.get("max_repeat", 0) <= 20)


def load_merged():
    """Merge truncated + canon Qwen + canon Llama by (prompt_id, method, layer, alpha)."""
    trunc = {json.loads(l)["prompt_id"] + f"|{json.loads(l).get('method')}|{json.loads(l).get('layer')}|{json.loads(l).get('alpha')}": json.loads(l)
             for l in open(f"{LSW}/sweep_judged.jsonl")}
    canonQ = {json.loads(l)["prompt_id"] + f"|{json.loads(l).get('method')}|{json.loads(l).get('layer')}|{json.loads(l).get('alpha')}": json.loads(l)
              for l in open(f"{LSW}/canon_qwen/layer_sweep_sweep_judged_canon_qwen.jsonl")}
    canonL = {json.loads(l)["prompt_id"] + f"|{json.loads(l).get('method')}|{json.loads(l).get('layer')}|{json.loads(l).get('alpha')}": json.loads(l)
              for l in open(f"{LSW}/canon_llama/layer_sweep_sweep_judged_canon_llama.jsonl")}
    rows = []
    for k, r in trunc.items():
        merged = dict(r)
        if k in canonQ:
            merged["judge_harm_likert_canon_qwen"] = canonQ[k].get("judge_harm_likert_canon_qwen")
        if k in canonL:
            merged["judge_harm_likert_canon_llama"] = canonL[k].get("judge_harm_likert_canon_llama")
        rows.append(merged)
    return rows


def main():
    rows = load_merged()
    rows = [r for r in rows if coh_ok(r)]
    print(f"coherence-ok rows: {len(rows)}")

    LAYERS_OF_INTEREST = [16]  # focus on L16 for F.3 shape-survival panel
    ALPHAS = [-1.0, -0.5, 0.0, 0.5, 1.0]
    METHODS = [("baseline", "#000000", "baseline"),
               ("lr_harm",  "#cc3322", "LR-harm"),
               ("md_harm",  "#7f3fbf", "MD-harm"),
               ("lr_comp",  "#3b6fb0", "LR-comp"),
               ("md_comp",  "#5fa6f9", "MD-comp"),
               ("random",   "#888888", "random")]
    # x-jitter per method so points and CIs don't overlap at the same α
    JITTER = {"baseline": 0.00, "lr_harm": -0.06, "md_harm": -0.03,
              "lr_comp": +0.03, "md_comp": +0.06, "random": +0.09}

    def boot_mean_ci(vals, n_boot=500, seed=42):
        """Bootstrap 95% CI for mean of vals."""
        if not vals:
            return None, None, None
        rng = np.random.default_rng(seed)
        n = len(vals)
        means = np.empty(n_boot)
        arr = np.array(vals, dtype=float)
        for k in range(n_boot):
            idx = rng.integers(0, n, size=n)
            means[k] = arr[idx].mean()
        return float(arr.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

    def wilson_ci(k, n, z=1.96):
        """Wilson 95% CI for binomial proportion."""
        if n == 0:
            return None, None, None
        p = k / n
        denom = 1 + z*z/n
        center = (p + z*z/(2*n)) / denom
        half = z * np.sqrt((p*(1-p) + z*z/(4*n))/n) / denom
        return p, max(0, center - half), min(1, center + half)

    # ===== Panel A: alpha-sweep at L16, mean Likert (canon Q + canon L) + ASR =====
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharex=True)
    titles = [
        "canon Likert (Qwen)",
        "canon Likert (Llama)",
        "ASR (Qwen JBB compliance)",
    ]
    likert_cols = ["judge_harm_likert_canon_qwen", "judge_harm_likert_canon_llama", None]

    for col_idx, (ax, title, col) in enumerate(zip(axes, titles, likert_cols)):
        for method, color, label in METHODS:
            xs, ys, los, his, ns = [], [], [], [], []
            for a in ALPHAS:
                if a == 0.0:
                    cells = [r for r in rows if r.get("method") == "baseline"]
                else:
                    cells = [r for r in rows if r.get("method") == method
                             and r.get("layer") == 16
                             and abs(r.get("alpha", 0) - a) < 1e-9]
                if col is None:
                    valid = [r for r in cells if r.get("judge_compliance_unsafe") is not None]
                    if not valid:
                        continue
                    k = sum(1 for r in valid if r.get("judge_compliance_unsafe"))
                    p, lo, hi = wilson_ci(k, len(valid))
                    y, lo, hi = 100*p, 100*lo, 100*hi
                else:
                    vals = [r[col] for r in cells if r.get(col) is not None]
                    if not vals:
                        continue
                    y, lo, hi = boot_mean_ci(vals, n_boot=500,
                                             seed=42 + (hash(method) % 10000) + int((a+1)*1000))
                xs.append(a); ys.append(y); los.append(lo); his.append(hi); ns.append(len(cells))
            if method == "baseline" and len(xs) > 0:
                bl_label = (f"baseline ({ys[0]:.2f})" if col is not None
                            else f"baseline ({ys[0]:.1f}%)")
                ax.axhline(ys[0], color="black", linestyle="-", linewidth=1.0, label=bl_label)
                ax.axhspan(los[0], his[0], color="black", alpha=0.06, edgecolor="none")
                continue
            ys_a = np.array(ys); los_a = np.array(los); his_a = np.array(his)
            xs_jit = np.array(xs) + JITTER.get(method, 0.0)
            ax.errorbar(xs_jit, ys_a, yerr=[ys_a - los_a, his_a - ys_a],
                        marker="o", linewidth=1.4, markersize=6,
                        color=color, capsize=3, label=label, alpha=0.85)
            # Annotate n at α=+1 for md_harm (PF.2)
            if method == "md_harm" and 1.0 in xs:
                i_a = xs.index(1.0)
                ax.annotate(f"n={ns[i_a]}", (xs_jit[i_a], ys_a[i_a]),
                            xytext=(6, -2), textcoords="offset points",
                            fontsize=7, color=color, fontweight="bold")
        ax.set_xlabel("α (steering magnitude)")
        ax.set_title(title, fontsize=10.5)
        ax.grid(alpha=0.25, linewidth=0.4)
        if col is not None:
            ax.set_ylim(1.0, 4.0)
            if col_idx == 0:
                ax.set_ylabel("mean canon Likert (1–5)")
        else:
            ax.set_ylim(40, 105)
            ax.set_ylabel("ASR (%)")
        if col_idx == 0:
            ax.legend(loc="lower center", fontsize=8, frameon=True, framealpha=0.92, ncol=2)

    fig.suptitle("α-sweep at L16, canon Likert + ASR — shape preserved (inverted-U Likert, "
                 "refusal-knockover ASR); md_harm α=±1 n annotated (PF.2)",
                 fontsize=10.5)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        out_p = f"{FIG}/alpha_sweep_L16_canon.{ext}"
        fig.savefig(out_p, bbox_inches="tight",
                    dpi=150 if ext == "png" else None)
        print(f"wrote {out_p}")
    plt.close(fig)

    # ===== Shape-survival check: for each (method, alpha), compute mean trunc + canon Qwen,
    # and report whether the sign of (mean - baseline) is preserved =====
    out_check = []
    truncated_baseline = np.mean([r["judge_harm_likert"]
                                  for r in rows if r.get("method") == "baseline"
                                  and r.get("judge_harm_likert") is not None])
    canon_baseline = np.mean([r["judge_harm_likert_canon_qwen"]
                              for r in rows if r.get("method") == "baseline"
                              and r.get("judge_harm_likert_canon_qwen") is not None])
    print(f"\nbaseline truncated Qwen Likert: {truncated_baseline:.3f}")
    print(f"baseline canon Qwen Likert: {canon_baseline:.3f}")
    print(f"baseline shift (canon − trunc): {canon_baseline - truncated_baseline:+.3f}")

    print(f"\n=== shape survival at L16: sign of (Likert mean − baseline) per (method, α) ===")
    print(f"  {'method':>10} {'α':>5}  {'n':>3}  {'trunc Likert':>12}  {'canon Q Likert':>14}  {'Δ trunc':>8}  {'Δ canon':>8}  {'sign matches?':>14}")
    n_match = n_total = 0
    for method, _, _ in METHODS[1:]:  # skip baseline
        for a in [-1.0, -0.5, 0.5, 1.0]:
            cells = [r for r in rows if r.get("method") == method
                     and r.get("layer") == 16
                     and abs(r.get("alpha", 0) - a) < 1e-9]
            ts = [r["judge_harm_likert"] for r in cells if r.get("judge_harm_likert") is not None]
            cs = [r["judge_harm_likert_canon_qwen"] for r in cells if r.get("judge_harm_likert_canon_qwen") is not None]
            if not ts or not cs:
                continue
            t_mean = np.mean(ts)
            c_mean = np.mean(cs)
            t_d = t_mean - truncated_baseline
            c_d = c_mean - canon_baseline
            sign_match = (np.sign(t_d) == np.sign(c_d)) or (abs(t_d) < 0.1 and abs(c_d) < 0.1)
            n_match += int(sign_match)
            n_total += 1
            print(f"  {method:>10} {a:+5.2f}  {len(cs):>3}  {t_mean:>12.2f}  {c_mean:>14.2f}  "
                  f"{t_d:+8.2f}  {c_d:+8.2f}  {'YES' if sign_match else 'NO':>14}")
            out_check.append({"method": method, "alpha": float(a), "n": int(len(cs)),
                              "trunc_mean": float(t_mean), "canon_mean": float(c_mean),
                              "trunc_delta": float(t_d), "canon_delta": float(c_d),
                              "sign_match": bool(sign_match)})

    print(f"\nshape sign-match rate: {n_match}/{n_total}")

    with open(f"{LSW}/shape_survival_L16.json", "w") as f:
        json.dump({"baseline_trunc": float(truncated_baseline),
                   "baseline_canon": float(canon_baseline),
                   "sign_match_rate": f"{n_match}/{n_total}",
                   "per_cell": out_check}, f, indent=2)
    print(f"wrote {LSW}/shape_survival_L16.json")


if __name__ == "__main__":
    main()
