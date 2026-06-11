"""Loss-set α-sweep figure at L16, canon judges + Wilson CIs.

Built from:
  experiments/steering_v3/sweep_p2_judged.jsonl              (truncated, has prompt_type)
  experiments/steering_v3/canon_qwen/sweep_p2_judged_canon_qwen.jsonl
  experiments/steering_v3/canon_llama/sweep_p2_judged_canon_llama.jsonl
  experiments/steering_v3/sweep_random_L16_jbb_qwen.jsonl   (random control, JBB Qwen)
  experiments/steering_v3/sweep_random_L16_canon_qwen.jsonl (random control, canon Qwen)

Schema: sweep_p2 is single-axis α_c sweep at L16, α_h=0 throughout. Single
direction (compliance-direction steering). prompt_type ∈ {win, loss}.

Three panels (loss-set only):
  - ASR (Qwen JBB), Wilson 95% CIs, two lines: compliance-dir vs random
  - canon Qwen Likert, bootstrap CIs, both lines
  - canon Llama Likert, bootstrap CIs, compliance-dir only (random has no canon Llama)

x-jitter to separate overlapping points.
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/Users/julianquick/portfolio_copy/turnstile"
EXP = f"{ROOT}/experiments/steering_v3"
FIG = f"{ROOT}/figures"


def coh_ok(r):
    c = r.get("coherence", {}) or {}
    return (c.get("n_tokens", 0) >= 5 and
            c.get("token_unique_ratio", 1) >= 0.15 and
            c.get("max_repeat", 0) <= 20)


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return None, None, None
    p = k / n
    denom = 1 + z*z/n
    center = (p + z*z/(2*n)) / denom
    half = z * np.sqrt((p*(1-p) + z*z/(4*n))/n) / denom
    return p, max(0, center - half), min(1, center + half)


def boot_mean_ci(vals, n_boot=500, seed=42):
    if not vals:
        return None, None, None
    rng = np.random.default_rng(seed)
    arr = np.array(vals, dtype=float)
    n = len(arr)
    means = np.empty(n_boot)
    for k in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[k] = arr[idx].mean()
    return float(arr.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def load_keyed(path, key_fields=("prompt_id", "alpha_c", "alpha_h")):
    by_key = {}
    if not os.path.exists(path):
        return by_key
    for line in open(path):
        try:
            r = json.loads(line)
        except Exception:
            continue
        k = tuple(str(r.get(f, "")) for f in key_fields)
        by_key[k] = r
    return by_key


def main():
    trunc = load_keyed(f"{EXP}/sweep_p2_judged.jsonl")
    canonQ = load_keyed(f"{EXP}/canon_qwen/sweep_p2_judged_canon_qwen.jsonl")
    canonL = load_keyed(f"{EXP}/canon_llama/sweep_p2_judged_canon_llama.jsonl")
    rand_jbb = load_keyed(f"{EXP}/sweep_random_L16_jbb_qwen.jsonl")
    rand_canonQ = load_keyed(f"{EXP}/sweep_random_L16_canon_qwen.jsonl")
    # Harm arm uses (method, layer, alpha) schema — load and re-key to alpha_c-equivalent
    harm_raw = [json.loads(l) for l in open(f"{EXP}/loss_arms/sweep_l16_harm_loss_final.jsonl")]
    # Translate harm-arm schema to use alpha_c/alpha_h equivalents for the plot loop
    for r in harm_raw:
        r["alpha_c"] = float(r.get("alpha", 0.0))
        r["alpha_h"] = 0.0
        r["prompt_type"] = "loss"  # this run was prompt-type-filter=loss

    # Merge sweep_p2 cells: trunc + canonQ + canonL
    merged_p2 = []
    for k, r in trunc.items():
        m = dict(r)
        if k in canonQ:
            m["judge_harm_likert_canon_qwen"] = canonQ[k].get("judge_harm_likert_canon_qwen")
        if k in canonL:
            m["judge_harm_likert_canon_llama"] = canonL[k].get("judge_harm_likert_canon_llama")
        merged_p2.append(m)
    merged_rand = []
    for k, r in rand_jbb.items():
        m = dict(r)
        if k in rand_canonQ:
            m["judge_harm_likert_canon_qwen"] = rand_canonQ[k].get("judge_harm_likert_canon_qwen")
        merged_rand.append(m)

    loss_p2 = [r for r in merged_p2 if r.get("prompt_type") == "loss" and coh_ok(r)]
    loss_rand = [r for r in merged_rand if r.get("prompt_type") == "loss" and coh_ok(r)]
    # Harm arm split into LR-harm and MD-harm; the layer_sweep_steering script
    # emits baseline cells with method="baseline" / alpha=0 (one per prompt,
    # no steering applied), so attach those to BOTH harm arms as the α=0 point.
    loss_harm_baseline = [r for r in harm_raw if r.get("method") == "baseline" and coh_ok(r)]
    # Normalize the baseline cells so they look like α=0 cells for each arm
    for r in loss_harm_baseline:
        r["alpha_c"] = 0.0
    loss_lr_harm = ([r for r in harm_raw if r.get("method") == "lr_harm" and coh_ok(r)]
                    + loss_harm_baseline)
    loss_md_harm = ([r for r in harm_raw if r.get("method") == "md_harm" and coh_ok(r)]
                    + loss_harm_baseline)
    print(f"loss-set comp-dir: {len(loss_p2)} rows")
    print(f"loss-set random:   {len(loss_rand)} rows")
    print(f"loss-set LR-harm:  {len(loss_lr_harm)} rows  (today, meta-llama)")
    print(f"loss-set MD-harm:  {len(loss_md_harm)} rows  (today, meta-llama)")
    print(f"loss-set harm baselines (today): {len(loss_harm_baseline)}")

    ALPHAS = [-1.0, -0.5, 0.0, 0.5, 1.0]
    JITTER = {"comp": -0.07, "random": -0.025, "lr_harm": +0.025, "md_harm": +0.07}
    arms = [
        ("comp",    "#3b6fb0", "compliance direction (May)",      loss_p2),
        ("random",  "#888888", "random direction (May)",          loss_rand),
        ("lr_harm", "#cc3322", "LR-harm direction (today)",       loss_lr_harm),
        ("md_harm", "#7f3fbf", "MD-harm direction (today)",       loss_md_harm),
    ]

    def cell(rows, alpha):
        return [r for r in rows if abs(r.get("alpha_c", 0) - alpha) < 1e-9 and r.get("alpha_h", 0) == 0]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), sharex=True)
    titles = ["ASR (Qwen JBB)", "canon Likert (Qwen)", "canon Likert (Llama)"]

    # ── Panel 0: ASR (all 4 arms, including α=0 per-arm baselines) ──
    ax = axes[0]
    for arm_id, color, label, rows in arms:
        xs, ys, los, his = [], [], [], []
        for a in ALPHAS:
            cells = cell(rows, a)
            comp = [r for r in cells if r.get("judge_compliance_unsafe") is not None]
            if not comp:
                continue
            k = sum(1 for r in comp if r.get("judge_compliance_unsafe"))
            p, lo, hi = wilson_ci(k, len(comp))
            xs.append(a + JITTER[arm_id])
            ys.append(p * 100); los.append(lo * 100); his.append(hi * 100)
        ys_a = np.array(ys); los_a = np.array(los); his_a = np.array(his)
        ax.errorbar(xs, ys_a, yerr=[ys_a - los_a, his_a - ys_a],
                    marker="o", linewidth=1.5, markersize=6, color=color,
                    capsize=3, label=label, alpha=0.9)
    ax.set_xticks(ALPHAS)
    ax.set_xlabel("α (steering magnitude, α_h=0)")
    ax.set_ylabel("ASR (%)")
    ax.set_title("ASR — loss set (Wilson 95% CI)", fontsize=10.5)
    ax.set_ylim(15, 100)
    ax.grid(alpha=0.25, linewidth=0.4)
    ax.legend(loc="lower right", fontsize=8, frameon=True, framealpha=0.92)

    # ── Panel 1+2: canon Likert (Qwen, Llama), all arms with α=0 included ──
    for pi, col_name in enumerate(["judge_harm_likert_canon_qwen",
                                    "judge_harm_likert_canon_llama"]):
        ax = axes[pi + 1]
        ax.axhline(4.0, color="#cc3322", linestyle=":", linewidth=1.0, alpha=0.6,
                   label="meaningful-uplift threshold (4)")
        for arm_id, color, label, rows in arms:
            if arm_id == "random" and pi == 1:
                continue  # random has no canon Llama
            xs, ys, los, his = [], [], [], []
            for a in ALPHAS:
                cells = cell(rows, a)
                vals = [r[col_name] for r in cells if r.get(col_name) is not None]
                if not vals:
                    continue
                m, lo, hi = boot_mean_ci(vals, n_boot=500,
                                         seed=42 + hash(arm_id + col_name) % 10000 + int((a+1)*1000))
                xs.append(a + JITTER[arm_id])
                ys.append(m); los.append(lo); his.append(hi)
            ys_a = np.array(ys); los_a = np.array(los); his_a = np.array(his)
            ax.errorbar(xs, ys_a, yerr=[ys_a - los_a, his_a - ys_a],
                        marker="o", linewidth=1.5, markersize=6, color=color,
                        capsize=3, label=label, alpha=0.9)
        ax.set_xticks(ALPHAS)
        ax.set_xlabel("α (steering magnitude, α_h=0)")
        ax.set_ylabel("mean canon Likert (1–5)")
        ax.set_title(titles[pi + 1], fontsize=10.5)
        ax.set_ylim(1.0, 4.2)
        ax.grid(alpha=0.25, linewidth=0.4)
        if pi == 0:
            ax.legend(loc="upper left", fontsize=8, frameon=True, framealpha=0.92)
        else:
            ax.legend(loc="upper left", fontsize=8, frameon=True, framealpha=0.92)

    fig.suptitle("Loss-set α-sweep at L16: ASR rises ~30pp with headroom (~45→~75%) but "
                 "canon Likert stays capped below meaningful-uplift (4) under every direction.\n"
                 "Comp + random arms generated May-26; harm arm generated today on the same "
                 "victim (meta-llama); per-arm α=0 baselines plotted alongside steered cells.",
                 fontsize=10.0)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        out_p = f"{FIG}/loss_set_alpha_sweep_L16.{ext}"
        fig.savefig(out_p, bbox_inches="tight",
                    dpi=150 if ext == "png" else None)
        print(f"wrote {out_p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
