"""Phase 4: plots and summary for the layer-sweep steering experiment.

Outputs:
  figures/layer_sweep_harm.pdf — Likert effect (vs baseline) vs layer, both
    harm directions + random as shaded band.
  figures/layer_sweep_compliance.pdf — same for compliance.
  figures/auc_vs_causal_by_layer.pdf — scatter of (probe AUC) vs
    (causal effect at α=−1.0), one point per (concept, method, layer).
  experiments/steering_v3/layer_sweep/summary.md — interpretive summary.
"""
import json
import math
from collections import defaultdict
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/Users/julianquick/portfolio_copy/turnstile"
LSW = f"{ROOT}/experiments/steering_v3/layer_sweep"
FIG = f"{ROOT}/figures"
LAYERS = [0, 4, 8, 12, 16, 20, 24, 28, 31]


def boot_mean(vals, B=500, seed=42):
    if not vals: return (float('nan'),)*3
    rng = np.random.default_rng(seed)
    ms = []
    for _ in range(B):
        idx = rng.integers(0, len(vals), size=len(vals))
        ms.append(np.mean([vals[i] for i in idx]))
    return float(np.mean(vals)), float(np.percentile(ms, 2.5)), float(np.percentile(ms, 97.5))


def coherence_ok(r):
    coh = r.get("coherence", {})
    if coh.get("n_tokens", 0) < 5:
        return False
    if coh.get("token_unique_ratio", 1.0) < 0.15:
        return False
    if coh.get("max_repeat", 0) > 20:
        return False
    return True


def main():
    rows = []
    n_degenerate = 0
    for line in open(f"{LSW}/sweep_judged.jsonl"):
        r = json.loads(line)
        if not coherence_ok(r):
            n_degenerate += 1
            continue
        rows.append(r)
    print(f"Loaded {len(rows)} rows (excluded {n_degenerate} degenerate)")

    meta = json.load(open(f"{LSW}/metadata.json"))

    # Baseline mean Likert + baseline ASR (layer-independent)
    bsl = [r for r in rows if r["method"] == "baseline"]
    bsl_harm = [r["judge_harm_likert"] for r in bsl if r["judge_harm_likert"] is not None]
    bsl_mean = float(np.mean(bsl_harm))
    bsl_comp = [r for r in bsl if r.get("judge_compliance_unsafe") is not None]
    bsl_asr = 100.0 * sum(1 for r in bsl_comp if r["judge_compliance_unsafe"]) / max(1, len(bsl_comp))
    print(f"Baseline (α=0, n={len(bsl_harm)}) mean Likert = {bsl_mean:.3f}, ASR = {bsl_asr:.1f}%")

    # Per (method, layer, alpha) collect harm Likert and ASR
    def per_cell(method, layer, alpha):
        rs = [r for r in rows if r["method"] == method
              and r["layer"] == layer
              and abs(r["alpha"] - alpha) < 1e-9
              and r["judge_harm_likert"] is not None]
        return rs

    # ===== Compute Δ_Likert (vs baseline) per (method, layer, α) =====
    def likert_effect(method, layer, alpha):
        rs = per_cell(method, layer, alpha)
        vals = [r["judge_harm_likert"] for r in rs]
        if not vals: return (float('nan'),)*3
        m, lo, hi = boot_mean(vals)
        return m - bsl_mean, lo - bsl_mean, hi - bsl_mean

    # ===== Plot 1 & 2: harm and compliance Δ_Likert vs layer =====
    def make_layer_plot(directions, title, outname):
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6), sharey=True)
        colors = {"lr": "#d65a31", "md": "#7f3fbf", "random": "#888888"}
        for col, alpha in enumerate([-1.0, 1.0]):
            ax = axes[col]
            for method, color, label in directions:
                xs, ys, lows, highs = [], [], [], []
                for L in LAYERS:
                    m, lo, hi = likert_effect(method, L, alpha)
                    xs.append(L); ys.append(m)
                    lows.append(lo); highs.append(hi)
                ys = np.array(ys); lows = np.array(lows); highs = np.array(highs)
                if "random" in method:
                    ax.fill_between(xs, lows, highs, color=color,
                                    alpha=0.20, label=label + " 95% CI",
                                    edgecolor="none")
                    ax.plot(xs, ys, color=color, linewidth=1.4,
                            linestyle="--", alpha=0.6)
                else:
                    ax.errorbar(xs, ys,
                                yerr=[ys - lows, highs - ys],
                                marker="o", markersize=6,
                                linewidth=1.6, color=color, capsize=3,
                                label=label)
            ax.axhline(0, color="black", linewidth=0.6)
            ax.set_xticks(LAYERS)
            ax.set_xticklabels([f"L{L}" for L in LAYERS], fontsize=9)
            ax.set_xlabel("steering layer")
            if col == 0:
                ax.set_ylabel(r"$\Delta$ mean harm Likert vs baseline (n=30 wins/cell)")
            ax.set_title(f"α = {alpha:+.1f}  (Δ vs baseline {bsl_mean:.2f})",
                         fontsize=10.5)
            ax.grid(alpha=0.25, linewidth=0.4)
            if col == 0:
                ax.legend(loc="lower right", fontsize=9, frameon=True,
                          framealpha=0.92)
        fig.suptitle(title, fontsize=11)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            out_p = f"{FIG}/{outname}.{ext}"
            fig.savefig(out_p, bbox_inches="tight",
                        dpi=150 if ext == "png" else None)
            print(f"wrote {out_p}")
        plt.close(fig)

    make_layer_plot(
        [("lr_harm", "#d65a31", "LR-harm direction"),
         ("md_harm", "#7f3fbf", "mean-diff harm direction"),
         ("random", "#888888", "random direction control")],
        "Causal effect of HARM directions on Stage-B Likert, by injection layer",
        "layer_sweep_harm")

    make_layer_plot(
        [("lr_comp", "#3b6fb0", "LR-compliance direction"),
         ("md_comp", "#5fa6f9", "mean-diff compliance direction"),
         ("random", "#888888", "random direction control")],
        "Causal effect of COMPLIANCE directions on Stage-B Likert, by injection layer",
        "layer_sweep_compliance")

    def likert_raw(method, layer, alpha):
        rs = [r for r in rows if r["method"] == method
              and r["layer"] == layer
              and abs(r["alpha"] - alpha) < 1e-9
              and r["judge_harm_likert"] is not None]
        if not rs:
            return (float('nan'),) * 3
        return boot_mean([r["judge_harm_likert"] for r in rs])

    # ===== ASR (binary compliance) plots, raw ASR with baseline line =====
    def asr_raw(method, layer, alpha):
        rs = [r for r in rows if r["method"] == method
              and r["layer"] == layer
              and abs(r["alpha"] - alpha) < 1e-9
              and r.get("judge_compliance_unsafe") is not None]
        if not rs:
            return (float('nan'),)*3
        bits = [1.0 if r["judge_compliance_unsafe"] else 0.0 for r in rs]
        m, lo, hi = boot_mean(bits)
        return 100*m, 100*lo, 100*hi

    def make_asr_plot(directions, title, outname):
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6), sharey=True)
        for col, alpha in enumerate([-1.0, 1.0]):
            ax = axes[col]
            for method, color, label in directions:
                xs, ys, lows, highs = [], [], [], []
                for L in LAYERS:
                    m, lo, hi = asr_raw(method, L, alpha)
                    xs.append(L); ys.append(m)
                    lows.append(lo); highs.append(hi)
                ys = np.array(ys); lows = np.array(lows); highs = np.array(highs)
                if "random" in method:
                    ax.fill_between(xs, lows, highs, color=color,
                                    alpha=0.20, label=label + " 95% CI",
                                    edgecolor="none")
                    ax.plot(xs, ys, color=color, linewidth=1.4,
                            linestyle="--", alpha=0.6)
                else:
                    ax.errorbar(xs, ys,
                                yerr=[ys - lows, highs - ys],
                                marker="o", markersize=6,
                                linewidth=1.6, color=color, capsize=3,
                                label=label)
            ax.axhline(bsl_asr, color="black", linewidth=1.0,
                       linestyle="-", label=f"baseline ({bsl_asr:.1f}%)"
                       if col == 0 else None)
            ax.set_xticks(LAYERS)
            ax.set_xticklabels([f"L{L}" for L in LAYERS], fontsize=9)
            ax.set_xlabel("steering layer")
            if col == 0:
                ax.set_ylabel("ASR (binary compliance %, n=30 wins/cell)")
            ax.set_title(f"α = {alpha:+.1f}", fontsize=10.5)
            ax.set_ylim(0, 105)
            ax.grid(alpha=0.25, linewidth=0.4)
            if col == 0:
                ax.legend(loc="lower right", fontsize=9, frameon=True,
                          framealpha=0.92)
        fig.suptitle(title, fontsize=11)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            out_p = f"{FIG}/{outname}.{ext}"
            fig.savefig(out_p, bbox_inches="tight",
                        dpi=150 if ext == "png" else None)
            print(f"wrote {out_p}")
        plt.close(fig)

    make_asr_plot(
        [("lr_harm", "#d65a31", "LR-harm direction"),
         ("md_harm", "#7f3fbf", "mean-diff harm direction"),
         ("random", "#888888", "random direction control")],
        "Causal effect of HARM directions on ASR (binary compliance), by injection layer",
        "layer_sweep_harm_asr")

    make_asr_plot(
        [("lr_comp", "#3b6fb0", "LR-compliance direction"),
         ("md_comp", "#5fa6f9", "mean-diff compliance direction"),
         ("random", "#888888", "random direction control")],
        "Causal effect of COMPLIANCE directions on ASR (binary compliance), by injection layer",
        "layer_sweep_compliance_asr")

    # ===== Combined cross-effect plot: all 4 directions on both measures =====
    # 2 rows (Likert, ASR) × 2 cols (α=-1, +1). Each panel: all 4 directions.
    all_dirs = [
        ("lr_harm", "#d65a31", "LR-harm"),
        ("md_harm", "#7f3fbf", "MD-harm"),
        ("lr_comp", "#3b6fb0", "LR-comp"),
        ("md_comp", "#5fa6f9", "MD-comp"),
        ("random",  "#888888", "random"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0), sharex="col")
    for col, alpha in enumerate([-1.0, 1.0]):
        # Row 0: raw Likert (1-5 scale) with baseline line
        ax = axes[0, col]
        for method, color, label in all_dirs:
            xs, ys, lows, highs = [], [], [], []
            for L in LAYERS:
                m, lo, hi = likert_raw(method, L, alpha)
                xs.append(L); ys.append(m)
                lows.append(lo); highs.append(hi)
            ys = np.array(ys); lows = np.array(lows); highs = np.array(highs)
            if method == "random":
                ax.fill_between(xs, lows, highs, color=color, alpha=0.20,
                                label=label + " 95% CI", edgecolor="none")
                ax.plot(xs, ys, color=color, linewidth=1.4,
                        linestyle="--", alpha=0.6)
            else:
                ax.errorbar(xs, ys, yerr=[ys-lows, highs-ys],
                            marker="o", markersize=5, linewidth=1.4,
                            color=color, capsize=2.5, label=label)
        ax.axhline(bsl_mean, color="black", linewidth=1.0,
                   label=f"baseline ({bsl_mean:.2f})" if col == 0 else None)
        ax.set_title(f"α = {alpha:+.1f}", fontsize=11)
        ax.set_ylim(1.0, 5.0)
        if col == 0:
            ax.set_ylabel("harm Likert (1–5, raw)")
        ax.grid(alpha=0.25, linewidth=0.4)
        if col == 0:
            ax.legend(loc="lower right", fontsize=8, frameon=True,
                      framealpha=0.92, ncol=2)

        # Row 1: raw ASR (binary compliance)
        ax = axes[1, col]
        for method, color, label in all_dirs:
            xs, ys, lows, highs = [], [], [], []
            for L in LAYERS:
                m, lo, hi = asr_raw(method, L, alpha)
                xs.append(L); ys.append(m)
                lows.append(lo); highs.append(hi)
            ys = np.array(ys); lows = np.array(lows); highs = np.array(highs)
            if method == "random":
                ax.fill_between(xs, lows, highs, color=color, alpha=0.20,
                                edgecolor="none")
                ax.plot(xs, ys, color=color, linewidth=1.4,
                        linestyle="--", alpha=0.6)
            else:
                ax.errorbar(xs, ys, yerr=[ys-lows, highs-ys],
                            marker="o", markersize=5, linewidth=1.4,
                            color=color, capsize=2.5)
        ax.axhline(bsl_asr, color="black", linewidth=1.0,
                   label=f"baseline ({bsl_asr:.1f}%)")
        ax.set_xticks(LAYERS)
        ax.set_xticklabels([f"L{L}" for L in LAYERS], fontsize=9)
        ax.set_xlabel("steering layer")
        if col == 0:
            ax.set_ylabel("ASR (binary compliance %)")
        ax.set_ylim(0, 105)
        ax.grid(alpha=0.25, linewidth=0.4)
        if col == 0:
            ax.legend(loc="lower right", fontsize=8, frameon=True,
                      framealpha=0.92)

    fig.suptitle("Own-effect vs cross-effect: harm and compliance directions, "
                 "both measures × both signs", fontsize=11.5)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        out_p = f"{FIG}/layer_sweep_cross_effects.{ext}"
        fig.savefig(out_p, bbox_inches="tight",
                    dpi=150 if ext == "png" else None)
        print(f"wrote {out_p}")
    plt.close(fig)

    # =====================================================================
    # "Harm ≠ Compliance" plots:
    #
    # (A) Dissociation scatter: ΔASR (x) vs ΔLikert (y) at α=+1.
    #     If the two probes captured the same concept, all points would
    #     cluster in one quadrant. They don't.
    # (B) Geometric orthogonality: cos(v_harm, v_comp) by layer.
    # =====================================================================

    # ---- (A) Dissociation scatter ----
    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    method_styles = {
        "lr_harm": ("#d65a31", "o", "LR-harm",  "harm-probe directions"),
        "md_harm": ("#7f3fbf", "s", "MD-harm",  "harm-probe directions"),
        "lr_comp": ("#3b6fb0", "o", "LR-comp",  "compliance-probe directions"),
        "md_comp": ("#5fa6f9", "s", "MD-comp",  "compliance-probe directions"),
        "random":  ("#888888", "x", "random",   "random control"),
    }
    for method, (color, marker, label, _group) in method_styles.items():
        xs, ys = [], []
        for L in LAYERS:
            asr_m, _, _ = asr_raw(method, L, 1.0)
            d_asr = asr_m - bsl_asr
            d_lik, _, _ = likert_effect(method, L, 1.0)
            if math.isnan(d_asr) or math.isnan(d_lik):
                continue
            xs.append(d_asr); ys.append(d_lik)
        ax.scatter(xs, ys, s=85, color=color, marker=marker,
                   edgecolor="black", linewidth=0.5, alpha=0.85, label=label)
        # annotate the L16/L20/L24 points to give the reader handles
        for L_idx, L in enumerate(LAYERS):
            if L in (12, 16, 20):
                ax.annotate(f"L{L}", (xs[L_idx], ys[L_idx]),
                            xytext=(5, 4), textcoords="offset points",
                            fontsize=7, color=color, alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_xlabel(r"$\Delta$ ASR (pp) at α=+1.0   →   raises binary compliance")
    ax.set_ylabel(r"$\Delta$ Likert at α=+1.0   →   raises harm magnitude")
    ax.set_title("Dissociation: pushing on the 'harm' direction raises ASR but lowers Likert.\n"
                 "Pushing on the 'compliance' direction lowers Likert and barely moves ASR.",
                 fontsize=10.5)
    ax.grid(alpha=0.25, linewidth=0.4)
    ax.legend(loc="lower left", fontsize=9, frameon=True, framealpha=0.92)
    # shaded quadrant labels
    xlim = ax.get_xlim(); ylim = ax.get_ylim()
    ax.text(xlim[1]*0.95, ylim[1]*0.6, "↑ both\n(more compliant\n+ more harmful)",
            ha="right", va="top", fontsize=8, color="#888", style="italic")
    ax.text(xlim[1]*0.95, ylim[0]*0.95, "more compliant\nless harmful",
            ha="right", va="bottom", fontsize=8, color="#444", style="italic")
    ax.text(xlim[0]*0.95, ylim[0]*0.95, "less compliant\nless harmful",
            ha="left", va="bottom", fontsize=8, color="#888", style="italic")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        out_p = f"{FIG}/harm_vs_compliance_dissociation.{ext}"
        fig.savefig(out_p, bbox_inches="tight",
                    dpi=150 if ext == "png" else None)
        print(f"wrote {out_p}")
    plt.close(fig)

    # ===== Signed-axis test: α on x-axis at fixed layer =====
    # If the direction is a real signed (more-X ↔ less-X) axis, the line
    # should be monotone in α and pass through baseline at α=0.
    # A "coherence crater" looks like: flat through the middle, dives at one
    # or both ends.
    ALPHAS = [-1.0, -0.5, 0.0, 0.5, 1.0]

    def line_at_layer(method, layer, alpha, kind):
        """kind ∈ {'likert', 'asr'}. Returns (mean, lo, hi)."""
        if alpha == 0.0:
            # baseline applies to every method/layer
            if kind == "likert":
                return boot_mean(bsl_harm)
            else:
                bits = [1.0 if r["judge_compliance_unsafe"] else 0.0
                        for r in bsl_comp]
                m, lo, hi = boot_mean(bits)
                return 100*m, 100*lo, 100*hi
        if kind == "likert":
            return likert_raw(method, layer, alpha)
        else:
            return asr_raw(method, layer, alpha)

    def make_alpha_panel(layer, outname):
        fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), sharex=True)
        # rows: measure (Likert, ASR), cols: concept emphasis (harm, comp).
        # Every panel shows ALL 5 directions; the column distinction marks
        # which directions are "own-concept" (solid+marker) vs "cross-concept"
        # (dashed, smaller marker, fainter).
        method_meta = {
            "lr_harm": ("#d65a31", "LR-harm",  "harm"),
            "md_harm": ("#7f3fbf", "MD-harm",  "harm"),
            "lr_comp": ("#3b6fb0", "LR-comp",  "comp"),
            "md_comp": ("#5fa6f9", "MD-comp",  "comp"),
        }
        # x-jitter so identical values don't occlude
        jitter = {"lr_harm": -0.03, "md_harm": -0.01,
                  "lr_comp": +0.01, "md_comp": +0.03}
        for col, concept in enumerate(["harm", "comp"]):
            for row, kind in enumerate(["likert", "asr"]):
                ax = axes[row, col]
                # random band first so it's under the lines
                ys, lows, highs = [], [], []
                for a in ALPHAS:
                    m, lo, hi = line_at_layer("random", layer, a, kind)
                    ys.append(m); lows.append(lo); highs.append(hi)
                ys = np.array(ys); lows = np.array(lows); highs = np.array(highs)
                ax.fill_between(ALPHAS, lows, highs, color="#888888",
                                alpha=0.18, edgecolor="none",
                                label="random 95% CI")
                ax.plot(ALPHAS, ys, color="#888888", linewidth=1.3,
                        linestyle=":", alpha=0.7)
                # All four named directions
                for method, (color, label, m_concept) in method_meta.items():
                    own = (m_concept == concept)
                    ys, lows, highs = [], [], []
                    for a in ALPHAS:
                        m, lo, hi = line_at_layer(method, layer, a, kind)
                        ys.append(m); lows.append(lo); highs.append(hi)
                    ys = np.array(ys); lows = np.array(lows); highs = np.array(highs)
                    xs = np.array(ALPHAS) + jitter[method]
                    if own:
                        ax.errorbar(xs, ys, yerr=[ys-lows, highs-ys],
                                    marker="o", markersize=6, linewidth=1.7,
                                    color=color, capsize=3,
                                    label=label + " (own-concept)")
                    else:
                        ax.errorbar(xs, ys, yerr=[ys-lows, highs-ys],
                                    marker="s", markersize=4, linewidth=1.0,
                                    linestyle="--", color=color, alpha=0.55,
                                    capsize=2,
                                    label=label + " (cross)")
                # baseline reference + α=0 anchor
                if kind == "likert":
                    ax.axhline(bsl_mean, color="black", linewidth=1.0,
                               label=f"baseline ({bsl_mean:.2f})")
                    ax.set_ylim(1.0, 5.0)
                    if col == 0:
                        ax.set_ylabel("harm Likert (1–5, raw)")
                else:
                    ax.axhline(bsl_asr, color="black", linewidth=1.0,
                               label=f"baseline ({bsl_asr:.1f}%)")
                    ax.set_ylim(0, 105)
                    if col == 0:
                        ax.set_ylabel("ASR (binary compliance %)")
                ax.axvline(0, color="black", linewidth=0.5, alpha=0.4)
                ax.set_xticks(ALPHAS)
                if row == 1:
                    ax.set_xlabel("α (steering magnitude, signed)")
                ax.grid(alpha=0.25, linewidth=0.4)
                ax.set_title(f"emphasizing {concept.upper()}-direction → "
                             f"{'Likert' if kind=='likert' else 'ASR'}",
                             fontsize=10.5)
                if col == 0 and row == 0:
                    ax.legend(loc="lower right", fontsize=7.5, frameon=True,
                              framealpha=0.92, ncol=1)
        fig.suptitle(f"Signed-axis test at layer L{layer}: α-sweep at fixed layer.\n"
                     f"A real signed axis is monotone in α; a coherence "
                     f"crater is flat through α=0 and dives at one or both "
                     f"ends.", fontsize=11)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            out_p = f"{FIG}/{outname}.{ext}"
            fig.savefig(out_p, bbox_inches="tight",
                        dpi=150 if ext == "png" else None)
            print(f"wrote {out_p}")
        plt.close(fig)

    make_alpha_panel(layer=16, outname="alpha_sweep_L16")

    # Small-multiples: 4 representative layers
    def make_alpha_smallmultiples(layers, kind, outname, title):
        n = len(layers)
        fig, axes = plt.subplots(2, n, figsize=(3.6*n, 7.2), sharex=True,
                                 sharey="row")
        method_meta = {
            "lr_harm": ("#d65a31", "LR-harm",  "harm"),
            "md_harm": ("#7f3fbf", "MD-harm",  "harm"),
            "lr_comp": ("#3b6fb0", "LR-comp",  "comp"),
            "md_comp": ("#5fa6f9", "MD-comp",  "comp"),
        }
        jitter = {"lr_harm": -0.03, "md_harm": -0.01,
                  "lr_comp": +0.01, "md_comp": +0.03}
        for ci, L in enumerate(layers):
            for ri, concept in enumerate(["harm", "comp"]):
                ax = axes[ri, ci] if n > 1 else axes[ri]
                # random band first
                ys, lows, highs = [], [], []
                for a in ALPHAS:
                    m, lo, hi = line_at_layer("random", L, a, kind)
                    ys.append(m); lows.append(lo); highs.append(hi)
                ys = np.array(ys); lows = np.array(lows); highs = np.array(highs)
                ax.fill_between(ALPHAS, lows, highs, color="#888888",
                                alpha=0.18, edgecolor="none")
                ax.plot(ALPHAS, ys, color="#888888", linewidth=1.1,
                        linestyle=":", alpha=0.7,
                        label="random" if (ri == 0 and ci == 0) else None)
                # All 4 named directions; own-concept solid, cross dashed
                for method, (color, label, m_concept) in method_meta.items():
                    own = (m_concept == concept)
                    ys, lows, highs = [], [], []
                    for a in ALPHAS:
                        m, lo, hi = line_at_layer(method, L, a, kind)
                        ys.append(m); lows.append(lo); highs.append(hi)
                    ys = np.array(ys); lows = np.array(lows); highs = np.array(highs)
                    xs = np.array(ALPHAS) + jitter[method]
                    if own:
                        ax.errorbar(xs, ys, yerr=[ys-lows, highs-ys],
                                    marker="o", markersize=4.5, linewidth=1.4,
                                    color=color, capsize=2.5,
                                    label=label if (ri == 0 and ci == 0) else None)
                    else:
                        ax.errorbar(xs, ys, yerr=[ys-lows, highs-ys],
                                    marker="s", markersize=3.0, linewidth=0.9,
                                    linestyle="--", color=color, alpha=0.55,
                                    capsize=2,
                                    label=label + " (cross)"
                                    if (ri == 0 and ci == 0) else None)
                # baseline + zero line
                if kind == "likert":
                    bsl_y = bsl_mean
                    ax.set_ylim(1.0, 5.0)
                else:
                    bsl_y = bsl_asr
                    ax.set_ylim(0, 105)
                ax.axhline(bsl_y, color="black", linewidth=0.9,
                           label=("baseline" if (ri == 0 and ci == 0) else None))
                ax.axvline(0, color="black", linewidth=0.5, alpha=0.4)
                ax.set_xticks(ALPHAS)
                ax.grid(alpha=0.25, linewidth=0.4)
                if ri == 0:
                    ax.set_title(f"L{L}", fontsize=10.5)
                if ci == 0:
                    ax.set_ylabel(f"{concept.upper()}-dir emphasis →\n"
                                  f"{'Likert (1–5)' if kind=='likert' else 'ASR (%)'}")
                if ri == 1:
                    ax.set_xlabel("α")
        if n > 1:
            axes[0, 0].legend(loc="lower right", fontsize=7.5, frameon=True,
                              framealpha=0.92, ncol=1)
        else:
            axes[0].legend(loc="lower right", fontsize=7.5, frameon=True,
                           framealpha=0.92)
        fig.suptitle(title, fontsize=11)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            out_p = f"{FIG}/{outname}.{ext}"
            fig.savefig(out_p, bbox_inches="tight",
                        dpi=150 if ext == "png" else None)
            print(f"wrote {out_p}")
        plt.close(fig)

    rep_layers = [4, 12, 16, 24, 31]
    make_alpha_smallmultiples(rep_layers, "likert",
                              "alpha_sweep_smallmultiples_likert",
                              "α-sweep, raw Likert, across representative layers")
    make_alpha_smallmultiples(rep_layers, "asr",
                              "alpha_sweep_smallmultiples_asr",
                              "α-sweep, raw ASR, across representative layers")

    # ---- (B) Geometric orthogonality: cos(v_harm, v_comp) by layer ----
    import torch as _torch
    DIRS = f"{LSW}/directions"

    def _load(name):
        return _torch.load(f"{DIRS}/{name}.pt", weights_only=False).float().numpy()

    cos_lr_hc = []
    cos_md_hc = []
    cos_lr_md_harm = []
    cos_lr_md_comp = []
    for L in LAYERS:
        v_lr_h = _load(f"v_lr_harm_L{L}"); v_md_h = _load(f"v_md_harm_L{L}")
        v_lr_c = _load(f"v_lr_comp_L{L}"); v_md_c = _load(f"v_md_comp_L{L}")
        cos_lr_hc.append(float(np.dot(v_lr_h, v_lr_c)))
        cos_md_hc.append(float(np.dot(v_md_h, v_md_c)))
        cos_lr_md_harm.append(float(np.dot(v_lr_h, v_md_h)))
        cos_lr_md_comp.append(float(np.dot(v_lr_c, v_md_c)))

    # Random-pair floor: expected |cos| of two random unit vectors in 4096-dim
    # is roughly 1/sqrt(4096) ≈ 0.0156 (1 SD).
    rand_sd = 1.0 / math.sqrt(4096)

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.axhspan(-rand_sd, rand_sd, color="#888", alpha=0.18,
               label=f"random-pair 1σ band (±{rand_sd:.3f})")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.plot(LAYERS, cos_lr_hc, marker="o", color="#cc3322", linewidth=1.8,
            label="cos(LR-harm, LR-comp)")
    ax.plot(LAYERS, cos_md_hc, marker="s", color="#a83fbf", linewidth=1.8,
            label="cos(MD-harm, MD-comp)")
    ax.plot(LAYERS, cos_lr_md_harm, marker="^", color="#444444", linewidth=1.2,
            linestyle="--", alpha=0.7, label="cos(LR-harm, MD-harm)  [same-concept consistency]")
    ax.plot(LAYERS, cos_lr_md_comp, marker="v", color="#777777", linewidth=1.2,
            linestyle="--", alpha=0.7, label="cos(LR-comp, MD-comp)  [same-concept consistency]")
    ax.set_xticks(LAYERS)
    ax.set_xticklabels([f"L{L}" for L in LAYERS], fontsize=9)
    ax.set_xlabel("layer")
    ax.set_ylabel("cosine similarity between direction vectors")
    ax.set_title("Harm and compliance directions are geometrically orthogonal.\n"
                 "Same-concept directions (dashed) agree across methods; cross-concept (solid) sits in the random-pair band.",
                 fontsize=10.5)
    ax.grid(alpha=0.25, linewidth=0.4)
    ax.legend(loc="upper left", fontsize=8.5, frameon=True, framealpha=0.92)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        out_p = f"{FIG}/harm_vs_compliance_orthogonality.{ext}"
        fig.savefig(out_p, bbox_inches="tight",
                    dpi=150 if ext == "png" else None)
        print(f"wrote {out_p}")
    plt.close(fig)

    # ===== Plot 3: AUC vs causal effect scatter =====
    # Causal effect = |Δ_Likert| at α=-1.0 minus random baseline at same α/layer
    fig, ax = plt.subplots(figsize=(8.0, 5.6))
    point_styles = {
        "lr_comp": ("#3b6fb0", "o", "LR-compliance"),
        "md_comp": ("#5fa6f9", "s", "MD-compliance"),
        "lr_harm": ("#d65a31", "o", "LR-harm"),
        "md_harm": ("#7f3fbf", "s", "MD-harm"),
    }
    auc_map = {
        "lr_comp": "lr_comp_auc", "md_comp": "md_comp_auc",
        "lr_harm": "lr_harm_auc", "md_harm": "md_harm_auc",
    }
    for method, (color, marker, label) in point_styles.items():
        xs, ys = [], []
        for L in LAYERS:
            auc = meta["per_layer"][f"L{L}"][auc_map[method]]
            m_eff, _, _ = likert_effect(method, L, -1.0)
            r_eff, _, _ = likert_effect("random", L, -1.0)
            net_effect = abs(m_eff) - abs(r_eff)  # excess over random baseline
            xs.append(auc); ys.append(net_effect)
        ax.scatter(xs, ys, color=color, marker=marker, s=70, alpha=0.85,
                   edgecolor="black", linewidth=0.5, label=label)
        # annotate with layer
        for i, L in enumerate(LAYERS):
            ax.annotate(f"L{L}", (xs[i], ys[i]),
                        xytext=(5, 3), textcoords="offset points", fontsize=7,
                        color=color, alpha=0.7)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xlabel("Probe AUC at this layer (cross-validated)")
    ax.set_ylabel(r"|$\Delta$ Likert| at α=−1.0  minus  |$\Delta$ Likert| of random direction")
    ax.set_title("Probe AUC (predictive) vs steering effect over random (causal)",
                 fontsize=10.5)
    ax.grid(alpha=0.25, linewidth=0.4)
    ax.legend(loc="upper left", fontsize=9, frameon=True, framealpha=0.92)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        out_p = f"{FIG}/auc_vs_causal_by_layer.{ext}"
        fig.savefig(out_p, bbox_inches="tight",
                    dpi=150 if ext == "png" else None)
        print(f"wrote {out_p}")
    plt.close(fig)

    # ===== Summary markdown =====
    md = []
    md.append("# Layer-sweep steering: summary")
    md.append("")
    md.append(f"- Total cells judged (after dropping {n_degenerate} degenerate): {len(rows)}")
    md.append(f"- Baseline mean Likert (α=0, n={len(bsl_harm)} wins): **{bsl_mean:.2f}**")
    md.append("")
    md.append("## Causal effect at α=−1.0 (Δ Likert vs baseline; suppression)")
    md.append("")
    md.append("| Layer | LR-harm Δ | MD-harm Δ | LR-comp Δ | MD-comp Δ | Random Δ |")
    md.append("|---|---|---|---|---|---|")
    for L in LAYERS:
        cells = []
        for m in ["lr_harm", "md_harm", "lr_comp", "md_comp", "random"]:
            e, _, _ = likert_effect(m, L, -1.0)
            cells.append(f"{e:+.2f}")
        md.append(f"| L{L} | {' | '.join(cells)} |")
    md.append("")
    md.append("## Causal effect at α=+1.0 (Δ Likert vs baseline; amplification)")
    md.append("")
    md.append("| Layer | LR-harm Δ | MD-harm Δ | LR-comp Δ | MD-comp Δ | Random Δ |")
    md.append("|---|---|---|---|---|---|")
    for L in LAYERS:
        cells = []
        for m in ["lr_harm", "md_harm", "lr_comp", "md_comp", "random"]:
            e, _, _ = likert_effect(m, L, 1.0)
            cells.append(f"{e:+.2f}")
        md.append(f"| L{L} | {' | '.join(cells)} |")
    md.append("")
    md.append(f"## ASR (binary compliance), baseline = {bsl_asr:.1f}%")
    md.append("")
    md.append("### α=−1.0 (raw ASR %; suppression)")
    md.append("")
    md.append("| Layer | LR-harm | MD-harm | LR-comp | MD-comp | Random |")
    md.append("|---|---|---|---|---|---|")
    for L in LAYERS:
        cells = []
        for m in ["lr_harm", "md_harm", "lr_comp", "md_comp", "random"]:
            e, _, _ = asr_raw(m, L, -1.0)
            cells.append(f"{e:.1f}%")
        md.append(f"| L{L} | {' | '.join(cells)} |")
    md.append("")
    md.append("### α=+1.0 (raw ASR %; amplification)")
    md.append("")
    md.append("| Layer | LR-harm | MD-harm | LR-comp | MD-comp | Random |")
    md.append("|---|---|---|---|---|---|")
    for L in LAYERS:
        cells = []
        for m in ["lr_harm", "md_harm", "lr_comp", "md_comp", "random"]:
            e, _, _ = asr_raw(m, L, 1.0)
            cells.append(f"{e:.1f}%")
        md.append(f"| L{L} | {' | '.join(cells)} |")
    md.append("")
    md.append("## Probe AUC by layer (cross-validated)")
    md.append("")
    md.append("| Layer | h_norm | LR-comp | MD-comp | LR-harm | MD-harm |")
    md.append("|---|---|---|---|---|---|")
    for L in LAYERS:
        p = meta["per_layer"][f"L{L}"]
        md.append(f"| L{L} | {p['h_norm_median']:.2f} | {p['lr_comp_auc']:.3f} | "
                  f"{p['md_comp_auc']:.3f} | {p['lr_harm_auc']:.3f} | {p['md_harm_auc']:.3f} |")

    out_text = "\n".join(md)
    with open(f"{LSW}/summary.md", "w") as f:
        f.write(out_text)
    print(f"\nWrote {LSW}/summary.md")
    print()
    print(out_text)


if __name__ == "__main__":
    main()
