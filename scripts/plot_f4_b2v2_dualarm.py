"""F.4 — B.2 v2 dual-arm robustness figure.

Reads from experiments/steering_v3/b2_v2/ (both arms generated TODAY on
meta-llama with identical stack; baselines 30/30 byte-identical between arms).

Plots ASR and canon Likert across α for both arms (Qwen-fit vs Llama-fit harm
direction at L16). Apply PF.1 (capped not flat) and PF.2 (annotate md α=±1 n).
Caption: dissociation robust to direction-fitting judge; Llama-fit modestly
weaker (PF.3).
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/Users/julianquick/portfolio_copy/turnstile"
B2V2 = f"{ROOT}/experiments/steering_v3/b2_v2"
FIG = f"{ROOT}/figures"


def coh_ok(r):
    c = r.get("coherence", {}) or {}
    return (c.get("n_tokens", 0) >= 5 and
            c.get("token_unique_ratio", 1) >= 0.15 and
            c.get("max_repeat", 0) <= 20)


def aggregate(arm, method, alpha, likert_col):
    cells = [r for r in arm if r.get("method") == method
             and r.get("layer") == 16
             and abs(r.get("alpha", 0) - alpha) < 1e-9
             and coh_ok(r)]
    if not cells:
        return None, None, None, 0
    comp = [r for r in cells if r.get("judge_compliance_unsafe") is not None]
    asr = 100 * sum(1 for r in comp if r.get("judge_compliance_unsafe")) / max(1, len(comp))
    lik = [r[likert_col] for r in cells if r.get(likert_col) is not None]
    lik_mean = np.mean(lik) if lik else None
    return asr, lik_mean, len(cells), len(cells)


def main():
    qwen_arm = [json.loads(l) for l in open(f"{B2V2}/sweep_l16_qwen_harm_today_final.jsonl")]
    llama_arm = [json.loads(l) for l in open(f"{B2V2}/sweep_l16_llama_harm_today_final.jsonl")]
    # Baseline (α=0) — same for both arms by construction; use Qwen arm
    bsl = [r for r in qwen_arm if r.get("method") == "baseline" and coh_ok(r)]
    bsl_comp = [r for r in bsl if r.get("judge_compliance_unsafe") is not None]
    bsl_asr = 100 * sum(1 for r in bsl_comp if r.get("judge_compliance_unsafe")) / max(1, len(bsl_comp))
    bsl_lik_q = np.mean([r["judge_harm_likert_canon_qwen"] for r in bsl
                         if r.get("judge_harm_likert_canon_qwen") is not None])
    bsl_lik_l = np.mean([r["judge_harm_likert_canon_llama"] for r in bsl
                         if r.get("judge_harm_likert_canon_llama") is not None])
    print(f"baseline ASR {bsl_asr:.1f}%  canon Qwen Likert {bsl_lik_q:.2f}  canon Llama Likert {bsl_lik_l:.2f}")

    ALPHAS = [-1.0, -0.5, 0.5, 1.0]
    METHODS = ["lr_harm", "md_harm"]

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.0), sharex=True)
    cols = [
        ("ASR (Qwen JBB)", None, "ASR (%)", 50, 105, bsl_asr),
        ("canon Likert (Qwen)", "judge_harm_likert_canon_qwen", "mean canon Likert (1-5)",
         1.0, 4.0, bsl_lik_q),
        ("canon Likert (Llama)", "judge_harm_likert_canon_llama", "mean canon Likert (1-5)",
         1.0, 4.0, bsl_lik_l),
    ]

    arm_styles = {
        "Qwen-fit": {"color": "#cc3322", "marker": "o", "linestyle": "-"},
        "Llama-fit": {"color": "#3b6fb0", "marker": "s", "linestyle": "--"},
    }

    for row_i, method in enumerate(METHODS):
        for col_i, (title, col_name, ylab, ymin, ymax, baseline_val) in enumerate(cols):
            ax = axes[row_i, col_i]
            for arm_name, arm in [("Qwen-fit", qwen_arm), ("Llama-fit", llama_arm)]:
                xs, ys, ns = [], [], []
                for a in ALPHAS:
                    if col_name is None:
                        # ASR
                        cells = [r for r in arm if r.get("method") == method
                                 and r.get("layer") == 16
                                 and abs(r.get("alpha", 0) - a) < 1e-9
                                 and coh_ok(r)]
                        comp = [r for r in cells if r.get("judge_compliance_unsafe") is not None]
                        if not comp:
                            continue
                        y = 100 * sum(1 for r in comp if r.get("judge_compliance_unsafe")) / len(comp)
                    else:
                        cells = [r for r in arm if r.get("method") == method
                                 and r.get("layer") == 16
                                 and abs(r.get("alpha", 0) - a) < 1e-9
                                 and coh_ok(r)]
                        vals = [r[col_name] for r in cells if r.get(col_name) is not None]
                        if not vals:
                            continue
                        y = np.mean(vals)
                    xs.append(a)
                    ys.append(y)
                    ns.append(len(cells))
                sty = arm_styles[arm_name]
                ax.plot(xs, ys, marker=sty["marker"], color=sty["color"],
                        linestyle=sty["linestyle"], linewidth=1.7, markersize=7,
                        label=arm_name + (" arm" if row_i == 0 and col_i == 0 else ""))
                # PF.2 — annotate n for md_harm at |α|=1
                if method == "md_harm":
                    for xi, yi, ni in zip(xs, ys, ns):
                        if abs(xi) > 0.6:
                            ax.annotate(f"n={ni}", (xi, yi),
                                        xytext=(6, -2 if arm_name == "Qwen-fit" else 8),
                                        textcoords="offset points",
                                        fontsize=7, color=sty["color"], fontweight="bold")
            # Baseline reference line
            ax.axhline(baseline_val, color="black", linestyle="-", linewidth=1.0, alpha=0.6,
                       label=f"baseline ({baseline_val:.1f}{'%' if col_i==0 else ''})"
                       if row_i == 0 and col_i == 0 else None)
            if col_name is not None:
                ax.axhline(4.0, color="#cc3322", linestyle=":", linewidth=1.0, alpha=0.5,
                           label="meaningful-uplift threshold (4)"
                           if row_i == 0 and col_i == 1 else None)
            ax.set_xticks(ALPHAS)
            if row_i == 1:
                ax.set_xlabel("α (steering magnitude)")
            if col_i == 0:
                ax.set_ylabel(f"{method.upper()} —\n{ylab}")
            else:
                ax.set_ylabel(ylab)
            ax.set_title(f"{title}" if row_i == 0 else None, fontsize=10.5)
            ax.set_ylim(ymin, ymax)
            ax.grid(alpha=0.25, linewidth=0.4)
            if row_i == 0 and col_i == 0:
                ax.legend(loc="lower center", fontsize=8, frameon=True, framealpha=0.92, ncol=2)
            elif row_i == 0 and col_i == 1:
                ax.legend(loc="upper left", fontsize=8, frameon=True, framealpha=0.92)

    fig.suptitle("B.2 v2 robustness: dissociation (ASR moves, canon Likert capped below "
                 "meaningful-uplift) holds whether harm direction is Qwen-fit or Llama-fit.\n"
                 "Both arms generated today on meta-llama, baselines 30/30 byte-identical; "
                 "md α=±1 n annotated (PF.2).",
                 fontsize=10.5)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        out_p = f"{FIG}/b2v2_dualarm_robustness.{ext}"
        fig.savefig(out_p, bbox_inches="tight",
                    dpi=150 if ext == "png" else None)
        print(f"wrote {out_p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
