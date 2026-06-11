"""Within-category k-fold CV for the outcome probe, pooled across all experiments.

For each JBB harm category, train and evaluate a logistic probe using ONLY that
category's conversations (stratified k-fold). Pools hidden states + labels from
every experiment with pre-extracted L16 hidden states at turn 0.

Conditions auto-detected:
  - Frozen victim: no victim adapters trained (default set for paper)
  - Hardened victim: victim adapters exist (distribution shift, reported separately)

Usage:
  python scripts/within_category_cv.py                  # all pooled (default)
  python scripts/within_category_cv.py --frozen-only    # exclude hardened
  python scripts/within_category_cv.py --list           # print source inventory
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


EXP_ROOT = "experiments"
TURN_INDEX = 0
N_SPLITS = 5
SEED = 42
OUT_JSON = "experiments/outcome_probe_v1/within_category_cv_pooled.json"


def discover_sources(exp_root=EXP_ROOT):
    """Find every experiment dir with hidden_states/ AND rounds/ present.

    Classify as frozen vs hardened by presence of victim_adapters/.
    """
    sources = []
    for name in sorted(os.listdir(exp_root)):
        d = os.path.join(exp_root, name)
        if not os.path.isdir(d):
            continue
        hs_dir = os.path.join(d, "hidden_states")
        rounds_dir = os.path.join(d, "rounds")
        if not (os.path.isdir(hs_dir) and os.path.isdir(rounds_dir)):
            continue
        n_pt = sum(1 for f in os.listdir(hs_dir) if f.endswith(".pt"))
        if n_pt == 0:
            continue
        hardened = os.path.isdir(os.path.join(d, "victim_adapters"))
        sources.append({"name": name, "hs_dir": hs_dir, "rounds_dir": rounds_dir,
                        "n_rounds": n_pt, "hardened": hardened})
    return sources


def load_turn0_pooled(sources, turn=TURN_INDEX):
    """Aggregate (X, y, category, source) across all sources at a single turn."""
    per_cat_hs = defaultdict(list)
    per_cat_y = defaultdict(list)
    per_cat_src = defaultdict(list)
    src_counts = defaultdict(int)

    for src in sources:
        hs_files = sorted(f for f in os.listdir(src["hs_dir"])
                          if f.startswith("round_") and f.endswith(".pt"))
        for hs_file in hs_files:
            round_num = int(hs_file.replace("round_", "").replace(".pt", ""))
            data = torch.load(os.path.join(src["hs_dir"], hs_file),
                              weights_only=False)
            jsonl_path = os.path.join(src["rounds_dir"], f"round_{round_num}.jsonl")
            cats = []
            if os.path.exists(jsonl_path):
                with open(jsonl_path) as f:
                    for line in f:
                        cats.append(json.loads(line).get("category", "unknown"))
            for i, (hs_tensor, label) in enumerate(zip(data["hidden_states"],
                                                        data["labels"])):
                if turn >= hs_tensor.shape[0]:
                    continue
                cat = cats[i] if i < len(cats) else "unknown"
                per_cat_hs[cat].append(hs_tensor[turn].numpy())
                per_cat_y[cat].append(int(bool(label)))
                per_cat_src[cat].append(src["name"])
                src_counts[src["name"]] += 1

    pooled = {cat: (np.stack(per_cat_hs[cat]),
                    np.array(per_cat_y[cat]),
                    per_cat_src[cat])
              for cat in per_cat_hs}
    return pooled, dict(src_counts)


def within_cat_auc(X, y, n_splits=N_SPLITS, seed=SEED):
    if len(np.unique(y)) < 2:
        return None
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    k = min(n_splits, n_pos, n_neg)
    if k < 2:
        return None

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    aucs = []
    for tr, te in skf.split(X, y):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        clf = LogisticRegression(C=1.0, class_weight="balanced",
                                 max_iter=2000, solver="lbfgs")
        clf.fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
    if not aucs:
        return None
    return {"auc_mean": float(np.mean(aucs)),
            "auc_std": float(np.std(aucs)),
            "n_folds_used": len(aucs),
            "n": int(len(y)), "n_pos": n_pos, "n_neg": n_neg}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frozen-only", action="store_true",
                   help="Exclude hardened-victim runs (distribution shift).")
    p.add_argument("--list", action="store_true", help="Print sources and exit.")
    p.add_argument("--turn", type=int, default=TURN_INDEX,
                   help="Single turn to evaluate (default 0).")
    p.add_argument("--all-turns", action="store_true",
                   help="Sweep turns 0-4. Overrides --turn.")
    p.add_argument("--out", type=str, default=OUT_JSON)
    args = p.parse_args()

    sources = discover_sources()
    if args.frozen_only:
        sources = [s for s in sources if not s["hardened"]]

    print(f"Discovered {len(sources)} sources with hidden_states + rounds:")
    for s in sources:
        tag = "hardened" if s["hardened"] else "frozen"
        print(f"  [{tag}] {s['name']:<25s} ({s['n_rounds']} rounds)")
    if args.list:
        return

    data, src_counts = load_turn0_pooled(sources, turn=TURN_INDEX)
    total = sum(len(y) for _, y, _ in data.values())
    total_wins = sum(int(y.sum()) for _, y, _ in data.values())
    print(f"\nPooled {total} conversations ({total_wins} wins, "
          f"{total_wins/total:.1%} ASR) across {len(data)} categories "
          f"at turn {TURN_INDEX} (L16).\n")

    print(f"{'Source':<25s}  {'convs':>6s}")
    for s, n in sorted(src_counts.items()):
        print(f"  {s:<23s}  {n:>6d}")
    print()

    print(f"{'Category':<30s}  {'n':>5s} {'n+':>4s}  {'folds':>5s}   AUC (mean ± std)")
    print("-" * 72)

    results = {}
    for cat in sorted(data):
        X, y, srcs = data[cat]
        r = within_cat_auc(X, y)
        if r is None:
            print(f"{cat:<30s}  {len(y):>5d} {int(y.sum()):>4d}  "
                  f"{'--':>5s}   insufficient data")
            continue
        # Count source diversity per category
        r["n_sources"] = len(set(srcs))
        results[cat] = r
        print(f"{cat:<30s}  {r['n']:>5d} {r['n_pos']:>4d}  "
              f"{r['n_folds_used']:>5d}   "
              f"{r['auc_mean']:.3f} ± {r['auc_std']:.3f}")

    if results:
        aucs = [r["auc_mean"] for r in results.values()]
        ws = [r["n"] for r in results.values()]
        print("-" * 72)
        print(f"Macro mean   : {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
        print(f"Weighted mean: {np.average(aucs, weights=ws):.3f}")
        print(f"Range        : {min(aucs):.3f}  –  {max(aucs):.3f}")

    agg = {
        "turn": TURN_INDEX,
        "layer_note": "L16 hidden states (per outcome_probe.py pipeline)",
        "frozen_only": args.frozen_only,
        "n_splits_requested": N_SPLITS,
        "seed": SEED,
        "sources": [{"name": s["name"], "hardened": s["hardened"],
                     "n_rounds": s["n_rounds"],
                     "convs_contributed": src_counts.get(s["name"], 0)}
                    for s in sources],
        "total_convs": int(total),
        "total_wins": int(total_wins),
        "per_category": results,
        "macro_mean": float(np.mean([r["auc_mean"] for r in results.values()])) if results else None,
        "macro_std": float(np.std([r["auc_mean"] for r in results.values()])) if results else None,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(agg, f, indent=2)
    print(f"\nWrote {args.out}")

    # Plot
    if results:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cats = sorted(results, key=lambda c: results[c]["auc_mean"])
        means = [results[c]["auc_mean"] for c in cats]
        stds = [results[c]["auc_std"] for c in cats]
        ns = [results[c]["n"] for c in cats]
        n_pos = [results[c]["n_pos"] for c in cats]
        win_rates = [p / n for p, n in zip(n_pos, ns)]

        fig, ax = plt.subplots(figsize=(9, 5))
        y_pos = np.arange(len(cats))
        colors = ["#e45a3b" if m >= 0.70 else ("#6b7280" if m < 0.65 else "#2a9d8f")
                  for m in means]
        ax.barh(y_pos, means, xerr=stds, color=colors, edgecolor="white",
                capsize=3, alpha=0.85)
        ax.axvline(0.5, color="black", linestyle=":", linewidth=1, label="chance")
        ax.axvline(np.mean(means), color="#264653", linestyle="--", linewidth=1.5,
                   label=f"macro mean ({np.mean(means):.3f})")

        labels = [f"{c}\n(n={n}, {wr:.0%} ASR)"
                  for c, n, wr in zip(cats, ns, win_rates)]
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Within-category outcome prediction AUC (5-fold CV)")
        title = ("Within-Category Outcome Prediction — "
                 f"pooled across {len(sources)} experiments "
                 f"({'frozen only' if args.frozen_only else 'all'}, {total:,} convs)")
        ax.set_title(title, fontsize=10)
        ax.set_xlim(0.48, 0.82)
        ax.legend(fontsize=9, loc="lower right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()

        out_png = args.out.replace(".json", ".png")
        out_pdf = args.out.replace(".json", ".pdf")
        plt.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.savefig(out_pdf, bbox_inches="tight")
        plt.close()
        print(f"Wrote {out_png}\nWrote {out_pdf}")


if __name__ == "__main__":
    main()
