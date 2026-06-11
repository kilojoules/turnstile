"""F.1 — Rebuilt cross-concept orthogonality figure.

Uses the N=1000 partition-shuffle null (with both directions refit per draw)
from null_cross_concept_cosines_n1000.json. Two panels side-by-side:
  - LR panel: LR/Qwen + LR/Llama observed cosines; band = per-layer null SD;
    ±0.03 detectable-cosine floor annotated.
  - MD panel: MD/Qwen + MD/Llama; band = per-layer null SD; ±0.13 floor.

The retraction must be visible:
  - LR/Qwen lights up *above* the null band at L4–L20 (real verdicts).
  - LR/Llama sits inside the band at L8+ (null).
  - Both MD lines sit inside their (wider) band — underpowered, not orthogonal.
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/Users/julianquick/portfolio_copy/turnstile"
NULL = f"{ROOT}/experiments/steering_v3/layer_sweep/null_cross_concept_cosines_n1000.json"
FIG = f"{ROOT}/figures"
LAYERS = [0, 4, 8, 12, 16, 20, 24, 28, 31]


def main():
    j = json.load(open(NULL))
    cells = j["per_cell"]

    def get(method, judge):
        rows = sorted([c for c in cells if c["method"] == method and c["judge"] == judge],
                      key=lambda c: c["layer"])
        obs = np.array([c["observed_cos"] for c in rows])
        sd = np.array([c["null_sd"] for c in rows])
        pct = np.array([c["percentile_in_null"] for c in rows])
        verd = [c["verdict"] for c in rows]
        return obs, sd, pct, verd

    lr_q_obs, lr_q_sd, lr_q_pct, lr_q_v = get("lr", "qwen")
    lr_l_obs, lr_l_sd, lr_l_pct, lr_l_v = get("lr", "llama")
    md_q_obs, md_q_sd, md_q_pct, md_q_v = get("md", "qwen")
    md_l_obs, md_l_sd, md_l_pct, md_l_v = get("md", "llama")

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0), sharey=False)
    x = np.array(LAYERS)

    # ── LR panel ──
    ax = axes[0]
    # Null band = ±2 × null SD (covers ~95% of null draws under normality)
    # Use the *Llama* SD for the band (slightly tighter) — they're nearly identical
    lr_band_sd = (lr_q_sd + lr_l_sd) / 2.0
    ax.fill_between(x, -2 * lr_band_sd, 2 * lr_band_sd, color="#888888", alpha=0.18,
                    label=r"partition-shuffle null ±2$\sigma$ (N=1000)", edgecolor="none")
    # ±0.03 detectable-cosine floor (one-sided 95% quantile of |null| at mid layers)
    ax.axhline(0.03, color="#444444", linestyle=":", linewidth=1.0,
               label="detectable-cosine floor ±0.03 (LR)")
    ax.axhline(-0.03, color="#444444", linestyle=":", linewidth=1.0)
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.4)

    # Qwen LR line — emphasize the "real" verdict cells at L4-L20
    ax.plot(x, lr_q_obs, marker="o", markersize=7, linewidth=1.7, color="#cc3322",
            label="LR-harm fit on Qwen labels")
    # Mark real cells with filled red, null with hollow
    for xi, yi, v in zip(x, lr_q_obs, lr_q_v):
        if v == "real":
            ax.plot(xi, yi, "o", markersize=9, color="#cc3322", markeredgecolor="black",
                    markeredgewidth=1.0)
        elif v == "borderline":
            ax.plot(xi, yi, "o", markersize=8, color="#ff9933", markeredgecolor="black",
                    markeredgewidth=0.6)
    # Llama LR line — should sit in band
    ax.plot(x, lr_l_obs, marker="s", markersize=7, linewidth=1.7, color="#3b6fb0",
            label="LR-harm fit on Llama labels")
    for xi, yi, v in zip(x, lr_l_obs, lr_l_v):
        if v == "real":
            ax.plot(xi, yi, "s", markersize=9, color="#3b6fb0", markeredgecolor="black",
                    markeredgewidth=1.0)
        elif v == "borderline":
            ax.plot(xi, yi, "s", markersize=8, color="#5b8fd0", markeredgecolor="black",
                    markeredgewidth=0.6)

    ax.set_xticks(LAYERS)
    ax.set_xticklabels([f"L{L}" for L in LAYERS], fontsize=9)
    ax.set_xlabel("residual-stream layer")
    ax.set_ylabel("cos(v_harm, v_comp)")
    ax.set_title("LR fit: Qwen-label orthogonality fails at L4–L20 under the proper null",
                 fontsize=10.5)
    ax.set_ylim(-0.06, 0.11)
    ax.grid(alpha=0.25, linewidth=0.4)
    ax.legend(loc="upper right", fontsize=8.5, frameon=True, framealpha=0.92)

    # ── MD panel ──
    ax = axes[1]
    md_band_sd = (md_q_sd + md_l_sd) / 2.0
    ax.fill_between(x, -2 * md_band_sd, 2 * md_band_sd, color="#888888", alpha=0.18,
                    label=r"partition-shuffle null ±2$\sigma$ (N=1000)", edgecolor="none")
    ax.axhline(0.13, color="#444444", linestyle=":", linewidth=1.0,
               label="detectable-cosine floor ±0.13 (MD)")
    ax.axhline(-0.13, color="#444444", linestyle=":", linewidth=1.0)
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.4)

    ax.plot(x, md_q_obs, marker="o", markersize=7, linewidth=1.7, color="#cc3322",
            label="MD-harm fit on Qwen labels")
    for xi, yi, v in zip(x, md_q_obs, md_q_v):
        if v == "real":
            ax.plot(xi, yi, "o", markersize=9, color="#cc3322", markeredgecolor="black",
                    markeredgewidth=1.0)
        elif v == "borderline":
            ax.plot(xi, yi, "o", markersize=8, color="#ff9933", markeredgecolor="black",
                    markeredgewidth=0.6)
    ax.plot(x, md_l_obs, marker="s", markersize=7, linewidth=1.7, color="#3b6fb0",
            label="MD-harm fit on Llama labels")
    for xi, yi, v in zip(x, md_l_obs, md_l_v):
        if v == "real":
            ax.plot(xi, yi, "s", markersize=9, color="#3b6fb0", markeredgecolor="black",
                    markeredgewidth=1.0)
        elif v == "borderline":
            ax.plot(xi, yi, "s", markersize=8, color="#5b8fd0", markeredgecolor="black",
                    markeredgewidth=0.6)

    ax.set_xticks(LAYERS)
    ax.set_xticklabels([f"L{L}" for L in LAYERS], fontsize=9)
    ax.set_xlabel("residual-stream layer")
    ax.set_ylabel("cos(v_harm, v_comp)")
    ax.set_title("MD fit: underpowered — null band is wide (±0.13), no verdicts above it",
                 fontsize=10.5)
    ax.set_ylim(-0.22, 0.22)
    ax.grid(alpha=0.25, linewidth=0.4)
    ax.legend(loc="upper left", fontsize=8.5, frameon=True, framealpha=0.92)

    fig.suptitle("Cross-concept orthogonality, N=1000 partition-shuffle null, both directions refit per draw",
                 fontsize=11)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        out_p = f"{FIG}/orthogonality_canon_n1000.{ext}"
        fig.savefig(out_p, bbox_inches="tight",
                    dpi=150 if ext == "png" else None)
        print(f"wrote {out_p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
