"""AUC vs layer for harm and compliance probes — post-audit values.

Sources: experiments/steering_v3/layer_sweep/metadata_llama.json which carries:
  - lr_comp_auc, md_comp_auc          (Llama-tagged per-turn breach, unchanged)
  - lr_harm_auc_qwen, md_harm_auc_qwen (Qwen Stage-B labels, canonical rubric)
  - lr_harm_auc_llama, md_harm_auc_llama (Llama Stage-B labels, canonical rubric)

Two-panel:
  Left: compliance probe AUC by layer (LR + MD).
  Right: harm probe AUC by layer (LR + MD, both Qwen and Llama label sources).
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/Users/julianquick/portfolio_copy/turnstile"
META = f"{ROOT}/experiments/steering_v3/layer_sweep/metadata_llama.json"
META_QWEN_COMP = f"{ROOT}/experiments/steering_v3/layer_sweep/metadata_qwen_comp.json"
FIG = f"{ROOT}/figures"
LAYERS = [0, 4, 8, 12, 16, 20, 24, 28, 31]


def main():
    m = json.load(open(META))["per_layer"]
    mq = json.load(open(META_QWEN_COMP))["per_layer"]
    x = np.array(LAYERS)

    def col(key):
        return np.array([m[f"L{L}"][key] for L in LAYERS])

    def col_q(key):
        return np.array([mq[f"L{L}"][key] for L in LAYERS])

    lr_comp = col("lr_comp_auc")
    md_comp = col("md_comp_auc")
    lr_comp_qwen = col_q("lr_comp_auc_qwen")
    md_comp_qwen = col_q("md_comp_auc_qwen")
    lr_h_q = col("lr_harm_auc_qwen")
    md_h_q = col("md_harm_auc_qwen")
    lr_h_l = col("lr_harm_auc_llama")
    md_h_l = col("md_harm_auc_llama")

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0), sharey=True)

    # ── Compliance panel ──
    ax = axes[0]
    ax.axhline(0.5, color="black", linestyle=":", linewidth=0.8, alpha=0.5,
               label="chance (0.5)")
    ax.plot(x, lr_comp, marker="o", markersize=7, linewidth=1.7, color="#cc3322",
            label="LR-comp (Llama labels)")
    ax.plot(x, lr_comp_qwen, marker="o", markersize=7, linewidth=1.7, color="#3b6fb0",
            label="LR-comp (Qwen labels)")
    ax.plot(x, md_comp, marker="s", markersize=7, linewidth=1.7, color="#cc3322",
            linestyle="--", alpha=0.7, label="MD-comp (Llama)")
    ax.plot(x, md_comp_qwen, marker="s", markersize=7, linewidth=1.7, color="#3b6fb0",
            linestyle="--", alpha=0.7, label="MD-comp (Qwen)")
    ax.set_xticks(LAYERS)
    ax.set_xticklabels([f"L{L}" for L in LAYERS], fontsize=9)
    ax.set_xlabel("residual-stream layer")
    ax.set_ylabel("5-fold CV AUC")
    ax.set_title("Compliance probe (per-turn breach labels, both judges)",
                 fontsize=10.5)
    ax.set_ylim(0.45, 0.85)
    ax.grid(alpha=0.25, linewidth=0.4)
    ax.legend(loc="lower right", fontsize=8.5, frameon=True, framealpha=0.92)

    # ── Harm panel ──
    ax = axes[1]
    ax.axhline(0.5, color="black", linestyle=":", linewidth=0.8, alpha=0.5)
    # Shade the L4-L20 mid-layer band where Llama-harm is materially weaker
    ax.fill_betweenx([0.45, 0.85], 3, 21, color="#cccccc", alpha=0.18, edgecolor="none",
                     label="L4–L20 (Llama-harm weaker; see B.2)")
    ax.plot(x, lr_h_q, marker="o", markersize=7, linewidth=1.7, color="#cc3322",
            label="LR-harm (Qwen Stage-B canon)")
    ax.plot(x, lr_h_l, marker="o", markersize=7, linewidth=1.7, color="#3b6fb0",
            label="LR-harm (Llama Stage-B canon)")
    ax.plot(x, md_h_q, marker="s", markersize=7, linewidth=1.7, color="#cc3322",
            linestyle="--", alpha=0.7, label="MD-harm (Qwen)")
    ax.plot(x, md_h_l, marker="s", markersize=7, linewidth=1.7, color="#3b6fb0",
            linestyle="--", alpha=0.7, label="MD-harm (Llama)")
    ax.set_xticks(LAYERS)
    ax.set_xticklabels([f"L{L}" for L in LAYERS], fontsize=9)
    ax.set_xlabel("residual-stream layer")
    ax.set_title("Harm probe (Stage-B Likert ≥ 4, canonical rubric, both judges)",
                 fontsize=10.5)
    ax.set_ylim(0.45, 0.85)
    ax.grid(alpha=0.25, linewidth=0.4)
    ax.legend(loc="lower right", fontsize=8.5, frameon=True, framealpha=0.92)

    fig.suptitle("Probe AUC by layer — post-audit (Llama-tagged compliance, "
                 "canon-rubric Stage-B for harm; Llama-harm AUC drops 0.05–0.10 at L4–L20)",
                 fontsize=10.5)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        out_p = f"{FIG}/auc_by_layer_postaudit.{ext}"
        fig.savefig(out_p, bbox_inches="tight",
                    dpi=150 if ext == "png" else None)
        print(f"wrote {out_p}")
    plt.close(fig)

    # Print a small table
    print("\nPer-layer table:")
    print(f"  {'L':>3}  {'LR-c L':>7}  {'LR-c Q':>7}  {'ΔLR-c':>7}  {'MD-c L':>7}  {'MD-c Q':>7}  "
          f"{'LR-h Q':>7}  {'LR-h L':>7}  {'ΔLR-h':>7}  {'MD-h Q':>7}  {'MD-h L':>7}")
    for i, L in enumerate(LAYERS):
        print(f"  L{L:<2}  {lr_comp[i]:>7.3f}  {lr_comp_qwen[i]:>7.3f}  "
              f"{lr_comp[i]-lr_comp_qwen[i]:+7.3f}  "
              f"{md_comp[i]:>7.3f}  {md_comp_qwen[i]:>7.3f}  "
              f"{lr_h_q[i]:>7.3f}  {lr_h_l[i]:>7.3f}  {lr_h_q[i]-lr_h_l[i]:+7.3f}  "
              f"{md_h_q[i]:>7.3f}  {md_h_l[i]:>7.3f}")
    mid_layers_i = [LAYERS.index(L) for L in [4, 8, 12, 16, 20]]
    print(f"\nMid-layer (L4–L20) means:")
    print(f"  LR-comp: Llama {np.mean(lr_comp[mid_layers_i]):.3f}, "
          f"Qwen {np.mean(lr_comp_qwen[mid_layers_i]):.3f}  "
          f"(Δ {np.mean(lr_comp[mid_layers_i]) - np.mean(lr_comp_qwen[mid_layers_i]):+.3f})")
    print(f"  LR-harm: Qwen {np.mean(lr_h_q[mid_layers_i]):.3f}, "
          f"Llama {np.mean(lr_h_l[mid_layers_i]):.3f}  "
          f"(Δ {np.mean(lr_h_q[mid_layers_i]) - np.mean(lr_h_l[mid_layers_i]):+.3f})")


if __name__ == "__main__":
    main()
