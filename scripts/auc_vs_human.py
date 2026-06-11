"""AUC of judge + regex + probe against human labels.

n is tiny (~10), so AUCs will be noisy. Reports with honest caveats.
"""
import json
import os
import re
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "scripts")
from regex_vs_probe_auc import (TAXONOMY, compile_taxonomy, score_text,
                                 load_pooled_hs_for_nethack)


def main():
    labels = [json.loads(l) for l in
              open("experiments/judge_calibration/labels.jsonl")]
    tasks_by_id = {}
    for line in open("experiments/judge_calibration/label_tasks.jsonl"):
        t = json.loads(line)
        tasks_by_id[t["id"]] = t

    compiled = compile_taxonomy()

    rows = []
    for r in labels:
        if r["human_label"] == "borderline":
            continue
        t = tasks_by_id[r["id"]]
        msgs = t["conversation"]
        last = next((m["content"] for m in reversed(msgs)
                     if m["role"] == "assistant"), "")
        cats, total, distinct = score_text(last, compiled)
        rows.append({
            "id": r["id"],
            "human_unsafe": 1 if r["human_label"] == "unsafe" else 0,
            "judge_unsafe": int(r["judge_unsafe"]),
            "regex_total": total,
            "regex_distinct": distinct,
            "regex_cats": list(cats.values()),
            "src": r["src"],
            "idx": t.get("_src_idx"),
        })

    y = np.array([r["human_unsafe"] for r in rows])
    judge = np.array([r["judge_unsafe"] for r in rows])
    total = np.array([r["regex_total"] for r in rows])
    distinct = np.array([r["regex_distinct"] for r in rows])

    n = len(y)
    print(f"n (excluding borderline): {n}", flush=True)
    print(f"human: unsafe={y.sum()}  safe={(y==0).sum()}", flush=True)
    print(f"judge: unsafe={judge.sum()}  safe={(judge==0).sum()}", flush=True)

    # Agreement
    agree = (y == judge).sum()
    print(f"\nJudge-Human agreement: {agree}/{n} = {agree/n:.1%}", flush=True)
    # Cohen's kappa manual
    po = agree / n
    pe = (y.mean() * judge.mean()) + ((1 - y.mean()) * (1 - judge.mean()))
    kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")
    print(f"Cohen's kappa: {kappa:.3f}", flush=True)

    if y.sum() == 0 or y.sum() == n:
        print("\n[no AUC — human labels all one class]", flush=True)
        return

    # AUCs vs human label
    print("\n=== AUC vs HUMAN label ===", flush=True)
    print(f"  Judge (binary):          AUC={roc_auc_score(y, judge):.3f}",
          flush=True)
    print(f"  Regex total_hits:        AUC={roc_auc_score(y, total):.3f}",
          flush=True)
    print(f"  Regex distinct_cats:     AUC={roc_auc_score(y, distinct):.3f}",
          flush=True)

    # per-row
    print("\nPer-row:", flush=True)
    print(f"{'id':>3s}  {'human':<10s}  {'judge':<6s}  "
          f"{'total':>5s}  {'dist':>4s}  src", flush=True)
    for r in rows:
        print(f"{r['id']:>3d}  {('unsafe' if r['human_unsafe'] else 'safe'):<10s}  "
              f"{str(bool(r['judge_unsafe'])):<6s}  "
              f"{r['regex_total']:>5d}  {r['regex_distinct']:>4d}  "
              f"{r['src']}", flush=True)

    # Confusion matrix
    print("\nConfusion (rows=human, cols=judge):", flush=True)
    cm = np.zeros((2, 2), dtype=int)
    for yi, ji in zip(y, judge):
        cm[yi, ji] += 1
    print(f"              judge=safe  judge=unsafe", flush=True)
    print(f"  human=safe      {cm[0,0]:>4d}         {cm[0,1]:>4d}", flush=True)
    print(f"  human=unsafe    {cm[1,0]:>4d}         {cm[1,1]:>4d}", flush=True)

    # FP rate / FN rate
    if cm[0].sum() > 0:
        fpr = cm[0, 1] / cm[0].sum()
        print(f"\nJudge FPR (flag safe as unsafe):  {fpr:.1%}", flush=True)
    if cm[1].sum() > 0:
        fnr = cm[1, 0] / cm[1].sum()
        print(f"Judge FNR (miss unsafe):          {fnr:.1%}", flush=True)


if __name__ == "__main__":
    main()
