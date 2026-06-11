"""Joint conversation-level bootstrap of per-category ASR and per-turn-label
compliance AUC.

For each category and each of B bootstrap iterations:
  1. Sample N_c conversations with replacement from that category's pool
     (N_c = total conversations in that category in the corpus).
  2. ASR_b = fraction of resampled conversations that breached.
  3. In-bag rows = per-turn rows from selected conversations (multiplicity
     preserved). OOB rows = per-turn rows from conversations not selected
     (each used once).
  4. Fit a fresh LogisticRegression on in-bag rows at the category's peak
     layer (from the existing StratifiedKFold run); evaluate AUC on OOB
     rows.

Both ASR and AUC are computed on the same resampled conversations, so
their uncertainties are jointly resampled and directly commensurate.

Outputs:
- experiments/outcome_probe_v1/joint_bootstrap_auc_asr.json
- figures/auc_vs_asr_scatter_bootstrap.{pdf,png}
"""
import gc
import glob
import json
import os
import time
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

POOLED_DIR = "/Users/julianquick/portfolio_copy/turnstile/experiments/pooled_hs"
OUT_DIR = "/Users/julianquick/portfolio_copy/turnstile/experiments/outcome_probe_v1"
FIG_DIR = "/Users/julianquick/portfolio_copy/turnstile/figures"

N_BOOT = 200
SEED = 42


def peak_layers_from_json(path):
    d = json.load(open(path))
    out = {}
    for cat, by_L in d["per_category_per_layer"].items():
        best = None
        for L, cell in by_L.items():
            if best is None or cell["auc"] > best["auc"]:
                best = {**cell, "layer": L}
        out[cat] = int(best["layer"].lstrip("L"))
    return out


def load_cat_data(layer):
    """Return per-conversation data at this layer:
       conv_X: list of per-turn rows arrays for each conversation
       conv_y: list of per-turn label arrays for each conversation
       conv_breach: list of bool (did the conversation breach)
       conv_cat: list of category strings
    """
    conv_X, conv_y, conv_breach, conv_cat = [], [], [], []
    for src in sorted(os.listdir(POOLED_DIR)):
        sdir = os.path.join(POOLED_DIR, src)
        if not os.path.isdir(sdir):
            continue
        for fn in sorted(os.listdir(sdir)):
            if not fn.endswith(".pt"):
                continue
            data = torch.load(os.path.join(sdir, fn), weights_only=False)
            arr = data["hidden_states_by_layer"][layer].numpy()
            labels = data["labels"].tolist()
            tobs = data["turns_of_breach"]
            cats = data["categories"]
            for i in range(len(labels)):
                breach = bool(labels[i])
                t_star = tobs[i] if breach else None
                if breach and t_star is None:
                    continue
                t_max = 4 if not breach else int(t_star)
                xs, ys = [], []
                for t in range(t_max + 1):
                    xs.append(arr[i, t])
                    ys.append(1 if (breach and t == t_star) else 0)
                conv_X.append(np.stack(xs))
                conv_y.append(np.array(ys))
                conv_breach.append(breach)
                conv_cat.append(cats[i])
            del data, arr
    gc.collect()
    return conv_X, conv_y, np.array(conv_breach), np.array(conv_cat)


def bootstrap_cat(conv_X, conv_y, conv_breach, B, seed):
    """Run B bootstrap iterations on one category's conversation pool.
    Returns (asr_array, auc_array)."""
    rng = np.random.default_rng(seed)
    n_conv = len(conv_X)
    asrs = []
    aucs = []
    for b in range(B):
        idx = rng.integers(0, n_conv, size=n_conv)  # with replacement
        inbag = np.zeros(n_conv, dtype=bool)
        inbag[idx] = True
        oob = ~inbag
        if oob.sum() < 2:
            continue

        # ASR on the resampled conversations
        breach_resample = conv_breach[idx]
        asr_b = breach_resample.mean()
        asrs.append(asr_b)

        # Build in-bag and OOB training/test matrices from per-turn rows
        Xtr = np.concatenate([conv_X[i] for i in idx], axis=0)
        ytr = np.concatenate([conv_y[i] for i in idx], axis=0)
        Xte = np.concatenate([conv_X[i] for i in np.where(oob)[0]], axis=0)
        yte = np.concatenate([conv_y[i] for i in np.where(oob)[0]], axis=0)

        if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
            continue

        clf = LogisticRegression(C=1.0, class_weight="balanced",
                                 max_iter=2000, solver="lbfgs")
        clf.fit(Xtr, ytr)
        aucs.append(roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]))

    return np.array(asrs), np.array(aucs)


def main():
    peak_layers = peak_layers_from_json(
        f"{OUT_DIR}/per_turn_label_per_category.json")
    cats = sorted(peak_layers.keys())

    # cache loaded data by layer (each category uses one layer; multiple
    # categories may share)
    cache = {}
    results = {}
    rng = np.random.default_rng(SEED)
    for ci, cat in enumerate(cats):
        L = peak_layers[cat]
        if L not in cache:
            t0 = time.time()
            print(f"\nLoading layer L{L} ...", flush=True)
            cache[L] = load_cat_data(L)
            print(f"  loaded in {time.time()-t0:.1f}s "
                  f"({len(cache[L][0])} convs)", flush=True)
        conv_X, conv_y, conv_breach, conv_cat = cache[L]
        mask = conv_cat == cat
        cX = [conv_X[i] for i in np.where(mask)[0]]
        cy = [conv_y[i] for i in np.where(mask)[0]]
        cB = conv_breach[mask]
        cat_seed = int(rng.integers(0, 2**31 - 1))
        t0 = time.time()
        asr_b, auc_b = bootstrap_cat(cX, cy, cB, N_BOOT, cat_seed)
        print(f"  [{ci+1}/{len(cats)}] {cat:<28}  L{L}  "
              f"n_conv={mask.sum()}  B={len(asr_b)}/{N_BOOT}  "
              f"({time.time()-t0:.1f}s)", flush=True)
        results[cat] = {
            "layer": L,
            "n_conv": int(mask.sum()),
            "asr_mean": float(asr_b.mean()),
            "asr_lo95": float(np.percentile(asr_b, 2.5)),
            "asr_hi95": float(np.percentile(asr_b, 97.5)),
            "auc_mean": float(auc_b.mean()),
            "auc_lo95": float(np.percentile(auc_b, 2.5)),
            "auc_hi95": float(np.percentile(auc_b, 97.5)),
            "n_boot_used": int(len(asr_b)),
            "asr_samples": asr_b.tolist(),
            "auc_samples": auc_b.tolist(),
        }

    out = {
        "n_boot_target": N_BOOT,
        "per_category": results,
        "seed": SEED,
        "splitter": "in-bag-vs-OOB conversation-level bootstrap "
                    "(no within-conversation leakage)",
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(f"{OUT_DIR}/joint_bootstrap_auc_asr.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_DIR}/joint_bootstrap_auc_asr.json")

    # ----- scatter plot with bootstrap CIs -----
    asrs = [results[c]["asr_mean"] * 100 for c in cats]
    asr_lo = [(results[c]["asr_mean"] - results[c]["asr_lo95"]) * 100
              for c in cats]
    asr_hi = [(results[c]["asr_hi95"] - results[c]["asr_mean"]) * 100
              for c in cats]
    aucs = [results[c]["auc_mean"] for c in cats]
    auc_lo = [results[c]["auc_mean"] - results[c]["auc_lo95"] for c in cats]
    auc_hi = [results[c]["auc_hi95"] - results[c]["auc_mean"] for c in cats]

    r_p, p_p = pearsonr(asrs, aucs)
    r_s, p_s = spearmanr(asrs, aucs)

    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    cmap = plt.get_cmap("tab10")
    ax.errorbar(asrs, aucs,
                xerr=[asr_lo, asr_hi], yerr=[auc_lo, auc_hi],
                fmt="none", ecolor="black", linewidth=0.8,
                capsize=2.5, alpha=0.85, zorder=2)
    for i, c in enumerate(cats):
        ax.scatter(asrs[i], aucs[i], s=80, color=cmap(i % 10),
                   edgecolor="black", linewidth=0.5, zorder=3, label=c)

    label_offsets = {
        "Fraud/Deception": (8, -2),
        "Malware/Hacking": (8, 2),
        "Privacy": (8, 0),
        "Government decision-making": (8, 0),
        "Expert advice": (-8, 8),
        "Economic harm": (-8, -10),
        "Physical harm": (8, 0),
        "Disinformation": (-8, 8),
        "Harassment/Discrimination": (-8, -10),
        "Sexual/Adult content": (8, 0),
    }
    for i, c in enumerate(cats):
        dx, dy = label_offsets.get(c, (8, 0))
        ax.annotate(c, (asrs[i], aucs[i]), xytext=(dx, dy),
                    textcoords="offset points", fontsize=8.5,
                    ha="left" if dx > 0 else "right", va="center")

    m, b = np.polyfit(asrs, aucs, 1)
    xs = np.linspace(min(asrs) - 2, max(asrs) + 2, 100)
    ax.plot(xs, m * xs + b, color="gray", linestyle="--", linewidth=0.9,
            alpha=0.7,
            label=f"linear fit (Pearson r={r_p:+.2f}, p={p_p:.2f}; "
                  f"Spearman ρ={r_s:+.2f}, p={p_s:.2f})")

    ax.set_xlabel("per-category ASR (%, conversation-level bootstrap 95% CI)")
    ax.set_ylabel("per-category compliance AUC (per-turn label, OOB, "
                  "same bootstrap 95% CI)")
    ax.set_title(f"Compliance AUC vs ASR by category\n"
                 f"(joint conversation-level bootstrap, B={N_BOOT}, no leakage)",
                 fontsize=10.5)
    ax.grid(alpha=0.25, linewidth=0.4)
    ax.legend(loc="lower right", fontsize=8.5, frameon=True,
              framealpha=0.93)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        out_p = f"{FIG_DIR}/auc_vs_asr_scatter_bootstrap.{ext}"
        fig.savefig(out_p, bbox_inches="tight", dpi=150 if ext == "png" else None)
        print(f"wrote {out_p}")
    plt.close(fig)

    print()
    print(f"Pearson r = {r_p:+.3f}, p = {p_p:.3f}")
    print(f"Spearman ρ = {r_s:+.3f}, p = {p_s:.3f}")
    print(f"{'category':<30}  {'ASR mean':>9}  {'ASR 95%':>15}    "
          f"{'AUC mean':>9}  {'AUC 95%':>15}")
    for c in sorted(cats, key=lambda c: -results[c]["auc_mean"]):
        r = results[c]
        print(f"  {c:<28}  {100*r['asr_mean']:>6.1f}%  "
              f"[{100*r['asr_lo95']:>4.1f}, {100*r['asr_hi95']:>4.1f}]   "
              f"{r['auc_mean']:>6.3f}  [{r['auc_lo95']:.3f}, {r['auc_hi95']:.3f}]")


if __name__ == "__main__":
    main()
