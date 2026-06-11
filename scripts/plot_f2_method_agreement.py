"""F.2 — Same-concept method-agreement bootstrap.

From C.2 bootstrap (n_iters=500), shows:
  - cos(LR-harm, MD-harm) by layer, for Qwen and Llama labels — tight + high
  - cos(LR-comp, MD-comp) by layer (Llama labels) — low and dropping with depth

The visual point: harm methods agree (CI excludes <0.7 at every layer); compliance
methods disagree (CI excludes >~0.4 at every layer past L0, drops with depth).
This is why we report LR for the orthogonality claim.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/Users/julianquick/portfolio_copy/turnstile"
BS = f"{ROOT}/experiments/steering_v3/layer_sweep/bootstrap_same_concept.json"
FIG = f"{ROOT}/figures"


def main():
    cells = json.load(open(BS))["per_cell"]
    LAYERS = sorted(set(c["layer"] for c in cells))
    x = np.array(LAYERS)

    def get(concept, judge):
        rows = sorted([c for c in cells if c["concept"] == concept and c["judge"] == judge],
                      key=lambda c: c["layer"])
        obs = np.array([c["observed"] for c in rows])
        lo = np.array([c["ci_low"] for c in rows])
        hi = np.array([c["ci_high"] for c in rows])
        return obs, lo, hi

    fig, ax = plt.subplots(figsize=(8.0, 5.0))

    # Harm — Qwen and Llama, both tight + high
    obs, lo, hi = get("harm", "qwen")
    ax.fill_between(x, lo, hi, color="#cc3322", alpha=0.18, edgecolor="none")
    ax.plot(x, obs, marker="o", markersize=7, linewidth=1.7, color="#cc3322",
            label="cos(LR-harm, MD-harm), Qwen labels")
    obs, lo, hi = get("harm", "llama")
    ax.fill_between(x, lo, hi, color="#7f3fbf", alpha=0.15, edgecolor="none")
    ax.plot(x, obs, marker="^", markersize=7, linewidth=1.7, color="#7f3fbf",
            label="cos(LR-harm, MD-harm), Llama labels")
    # Compliance — Llama labels only (labels for compliance are Llama-tagged regardless)
    obs, lo, hi = get("comp", "llama")
    ax.fill_between(x, lo, hi, color="#3b6fb0", alpha=0.20, edgecolor="none")
    ax.plot(x, obs, marker="s", markersize=7, linewidth=1.7, color="#3b6fb0",
            label="cos(LR-comp, MD-comp), Llama labels")

    ax.axhline(0.5, color="#444444", linestyle=":", linewidth=1.0, alpha=0.6,
               label="cos = 0.5 (reference)")
    ax.axhline(0.0, color="black", linewidth=0.5, alpha=0.4)
    ax.set_xticks(LAYERS)
    ax.set_xticklabels([f"L{L}" for L in LAYERS], fontsize=9)
    ax.set_xlabel("residual-stream layer")
    ax.set_ylabel("cos(LR, MD) on same concept, with 95% bootstrap CI (n=500 resamples)")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25, linewidth=0.4)
    ax.set_title("Methods agree on harm direction (CI > 0.7) but disagree on compliance "
                 "(CI < 0.4 past L0)",
                 fontsize=10.5)
    ax.legend(loc="lower left", fontsize=9, frameon=True, framealpha=0.92)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        out_p = f"{FIG}/method_agreement_canon.{ext}"
        fig.savefig(out_p, bbox_inches="tight",
                    dpi=150 if ext == "png" else None)
        print(f"wrote {out_p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
