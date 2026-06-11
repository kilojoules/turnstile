"""v2: Per-turn cascading-replay causal steering ablation.

Replaces the v1 pivot-anchored design (clamp_obliterated.py) which intervened
and judged only at the labeled breach turn t*. Two design changes:

1. Probe direction is fit on PER-TURN dual-judge labels (no t* anchor):
   refuse = turn judged dual-safe, comply = turn judged dual-unsafe.
2. Steering hook is active THROUGHOUT THE CONVERSATION (cascading replay).
   Every assistant turn is freshly generated under the intervention and
   judged with the strict dual judge.

Sharding: --records-start / --records-end let you run a slice of records
on each instance; probe directions are fit on the FULL dataset for
consistency across shards, but only the in-shard records' continuations
are generated and judged.

Usage (one shard at a time):
  python -m turnstile.clamp_obliterated_per_turn \\
      --replay-pt experiments/clamp_obliterated_v2/replay_v2_full.pt \\
      --output experiments/clamp_obliterated_v2_l16_shardA/ \\
      --probe-layer 16 --steer-layer 16 \\
      --records-start 0 --records-end 100 \\
      --alphas -8 -4 -2 0 2 4 8

Required record fields (besides v1 fields):
  ``original_per_turn_labels``: list of {turn: int, unsafe: bool|None}
                                from strict dual judge on original arm.
  ``original_hs_l{N}``         : tensor[n_turns, hidden] for N in {16, 31}.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from datetime import datetime, timezone

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from turnstile.model_utils import load_model, generate_text, unload_model
from turnstile.strict_dual_judge import StrictDualJudge, LocalJudgeFailure


# ---------------------------------------------------------------------------
# Steering hook (identical to v1)
# ---------------------------------------------------------------------------
class SteeringHook:
    def __init__(self, direction: torch.Tensor, alpha: float):
        self.direction = direction.detach().clone()
        self.alpha = float(alpha)
        self._handle = None

    def __call__(self, module, inputs, output):
        is_tuple = isinstance(output, tuple)
        hs = output[0] if is_tuple else output
        d = self.direction.to(hs.device, hs.dtype)
        hs = hs + self.alpha * d
        if is_tuple:
            return (hs,) + output[1:]
        return hs

    def register(self, model, layer_idx: int):
        layer = model.model.layers[layer_idx]
        self._handle = layer.register_forward_hook(self)
        return self

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


# ---------------------------------------------------------------------------
# Per-turn probe direction fitting (no t* anchor)
# ---------------------------------------------------------------------------
def _per_turn_xy(records, train_ids, hs_key, label_key):
    """Stack per-turn (hidden state, label) pairs from the original arm,
    over the records in train_ids. Skips turns without a usable label."""
    Xs, ys = [], []
    for i in train_ids:
        rec = records[i]
        hs = rec[hs_key]            # tensor [n_turns, hidden]
        labels = rec[label_key]     # list of {turn:int, unsafe:bool|None}
        label_by_turn = {l["turn"]: l for l in labels
                         if l.get("unsafe") is not None}
        n_turns = hs.shape[0]
        for t in range(n_turns):
            if t not in label_by_turn:
                continue
            Xs.append(hs[t].numpy())
            ys.append(1 if label_by_turn[t]["unsafe"] else 0)
    if not Xs:
        raise ValueError("no per-turn examples with labels in this fold's "
                         "training subset -- check label_key data")
    return np.stack(Xs), np.array(ys)


def fit_direction(records, train_ids, hs_key, label_key, seed=0,
                  method: str = "lr"):
    """Fit a per-turn compliance direction at the layer indicated by
    ``hs_key``. Returns (unit_direction, raw_norm, intercept, in_sample_auc).

    ``method``:
      "lr"       -- logistic regression coefficient (current default; picks up
                    distributed/subtle features but overfits on small N).
      "meandiff" -- Arditi-style mean(h | comply) - mean(h | refuse), the
                    natural geometric center-of-mass difference. Less
                    expressive but a faithful analogue to single-turn refusal-
                    direction work.
    """
    X, y = _per_turn_xy(records, train_ids, hs_key, label_key)
    if len(set(y.tolist())) < 2:
        raise ValueError(f"degenerate train fold: only label {set(y.tolist())}")

    if method == "lr":
        clf = LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=2000, solver="lbfgs",
            random_state=seed,
        )
        clf.fit(X, y)
        in_sample_auc = roc_auc_score(y, clf.decision_function(X))
        w = clf.coef_[0].astype(np.float32)
        intercept = float(clf.intercept_[0])
    elif method == "meandiff":
        mu_pos = X[y == 1].mean(axis=0)
        mu_neg = X[y == 0].mean(axis=0)
        w = (mu_pos - mu_neg).astype(np.float32)
        # In-sample AUC for the mean-diff direction: project and ROC.
        in_sample_auc = roc_auc_score(y, X @ w)
        intercept = 0.0
    else:
        raise ValueError(f"unknown direction method: {method!r}")

    direction = torch.from_numpy(w)
    raw = direction.norm().item()
    direction = direction / direction.norm()
    return direction, float(raw), float(intercept), float(in_sample_auc)


# ---------------------------------------------------------------------------
# Cascading-replay with steering hook active throughout
# ---------------------------------------------------------------------------
def replay_full_under_steering(vic_model, vic_tok, source_conv,
                                hook, layer_idx, max_tokens=300):
    """Replay each user turn of source_conv with hook active throughout.

    Produces FRESH assistant responses at every turn under the intervention.
    The user-side text comes from source_conv verbatim. Returns a list of
    {turn, response, conversation, oom} dicts (one per user turn).
    """
    n_user_turns = sum(1 for m in source_conv if m["role"] == "user")
    if hook is not None:
        hook.register(vic_model, layer_idx)
    outputs = []
    replayed = []
    try:
        for t in range(n_user_turns):
            user_msg = source_conv[2 * t]
            assert user_msg["role"] == "user", \
                f"expected user role at index {2*t}, got {user_msg['role']}"
            replayed.append(user_msg)
            prompt = vic_tok.apply_chat_template(
                replayed, tokenize=False, add_generation_prompt=True,
            )
            try:
                resp = generate_text(
                    vic_model, vic_tok, prompt,
                    max_tokens=max_tokens, temperature=0.0,
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                outputs.append({"turn": t, "response": None,
                                "oom": True, "conversation": list(replayed)})
                break
            replayed.append({"role": "assistant", "content": resp.strip()})
            outputs.append({"turn": t, "response": resp.strip(),
                            "oom": False, "conversation": list(replayed)})
        return outputs
    finally:
        if hook is not None:
            hook.remove()


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------
def run_sweep(args):
    os.makedirs(args.output, exist_ok=True)

    print(f"Loading {args.replay_pt}...")
    records = torch.load(args.replay_pt, weights_only=False)
    print(f"  {len(records)} records loaded (full dataset)")

    n_full = len(records)
    start = max(0, args.records_start)
    end = args.records_end if args.records_end > 0 else n_full
    end = min(end, n_full)
    print(f"  shard window: records[{start}:{end}] = {end - start} records")

    probe_layer = args.probe_layer
    steer_layer = args.steer_layer
    print(f"  probe_layer={probe_layer}  steer_layer={steer_layer}")

    if probe_layer == 16:
        hs_key = "original_hs"
    else:
        hs_key = f"original_hs_l{probe_layer}"
    label_key = "original_per_turn_labels"

    # Probe directions fit on FULL dataset (so all shards get the same
    # per-fold directions), but the sweep only runs on records in our shard.
    tobs = np.array([r["turn_of_breach"] for r in records])
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True,
                          random_state=args.seed)
    folds = list(skf.split(np.zeros(n_full), tobs))

    print(f"\n=== Phase 1: per-fold per-turn compliance direction at L{probe_layer} "
          f"(method={args.direction_method}) ===")
    fold_directions = []
    for k, (train_idx, test_idx) in enumerate(folds):
        direction, raw, intercept, in_auc = fit_direction(
            records, train_idx, hs_key, label_key, seed=args.seed + k,
            method=args.direction_method,
        )
        shard_test_ids = [int(i) for i in test_idx if start <= i < end]
        torch.save({
            "direction": direction,
            "raw_coef_norm": raw,
            "intercept": intercept,
            "in_sample_auc": in_auc,
            "train_record_ids": train_idx.tolist(),
            "test_record_ids_full": test_idx.tolist(),
            "shard_test_ids": shard_test_ids,
            "probe_layer": probe_layer,
            "steer_layer": steer_layer,
            "direction_method": args.direction_method,
        }, os.path.join(args.output, f"direction_fold{k}.pt"))
        fold_directions.append((direction, train_idx, shard_test_ids))
        print(f"  fold {k}: train={len(train_idx)}  shard_test={len(shard_test_ids)}  "
              f"raw||w||={raw:.3f}  intercept={intercept:+.3f}  in_auc={in_auc:.3f}")

    print(f"\nLoading victim {args.victim_model}...")
    vic_model, vic_tok = load_model(args.victim_model)
    print(f"Loading strict dual judge "
          f"(Llama-Guard-3-1B + Llama-3.1-70B-Instruct)...")
    judge = StrictDualJudge()

    alphas = list(args.alphas)
    arms = ["original", "obliterated"]
    print(f"\n=== Phase 2: per-turn cascading-replay sweep "
          f"alpha={alphas}  arms={arms} ===")

    results_path = os.path.join(args.output, "results.jsonl")
    seen = set()
    if os.path.exists(results_path):
        with open(results_path) as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                seen.add((row["source"], row["arm"], float(row["alpha"]),
                          int(row["turn"])))
        print(f"  resuming: {len(seen)} (record,arm,alpha,turn) tuples done")

    n_replays_planned = sum(len(t) for _, _, t in fold_directions) * len(alphas) * len(arms)
    print(f"  planned: {n_replays_planned} continuous replays "
          f"(~5 turn-judgments each)")

    t_start = time.time()
    n_replays_done = 0
    for k, (direction, _train_idx, shard_test_ids) in enumerate(fold_directions):
        for ridx in shard_test_ids:
            rec = records[ridx]
            for arm in arms:
                conv = (rec["original_conversation"] if arm == "original"
                        else rec["translated_conversation"])
                n_user_turns = sum(1 for m in conv if m["role"] == "user")
                for alpha in alphas:
                    expected = {(rec["source"], arm, float(alpha), t)
                                for t in range(n_user_turns)}
                    if expected.issubset(seen):
                        continue
                    hook = SteeringHook(direction, alpha) if alpha != 0 else None
                    try:
                        replays = replay_full_under_steering(
                            vic_model, vic_tok, conv,
                            hook=hook, layer_idx=steer_layer,
                            max_tokens=args.max_tokens,
                        )
                    except RuntimeError as e:
                        print(f"  [replay error] {rec['source']} arm={arm} "
                              f"α={alpha}: {e}")
                        if hook is not None:
                            hook.remove()
                        continue
                    for rep in replays:
                        turn = rep["turn"]
                        key = (rec["source"], arm, float(alpha), turn)
                        if key in seen:
                            continue
                        if rep["oom"] or rep["response"] is None:
                            row = {
                                "source": rec["source"], "fold": k,
                                "arm": arm, "alpha": float(alpha),
                                "turn": turn,
                                "probe_layer": probe_layer,
                                "steer_layer": steer_layer,
                                "direction_method": args.direction_method,
                                "turn_of_breach": rec["turn_of_breach"],
                                "goal": rec["goal"],
                                "response": None,
                                "verdict": {"unsafe": None, "reason": "oom"},
                                "judge_failed": True, "oom": True,
                                "ts": datetime.now(timezone.utc).isoformat(),
                            }
                            with open(results_path, "a") as f:
                                f.write(json.dumps(row) + "\n")
                            continue
                        try:
                            verdict = judge.judge_conversation(
                                rep["conversation"], goal=rec["goal"],
                            )
                            verdict_dict = asdict(verdict)
                            judge_failed = False
                        except LocalJudgeFailure as e:
                            print(f"  [judge fail] {rec['source']} arm={arm} "
                                  f"α={alpha} turn={turn}: {e}")
                            verdict_dict = {
                                "unsafe": None, "guard_unsafe": None,
                                "together_unsafe": None,
                                "raw_70b_output": str(e),
                                "reason": "judge_failed",
                            }
                            judge_failed = True
                        row = {
                            "source": rec["source"], "fold": k,
                            "arm": arm, "alpha": float(alpha),
                            "turn": turn,
                            "probe_layer": probe_layer,
                            "steer_layer": steer_layer,
                            "direction_method": args.direction_method,
                            "turn_of_breach": rec["turn_of_breach"],
                            "goal": rec["goal"],
                            "behavior": rec.get("behavior"),
                            "category": rec.get("category"),
                            "response": rep["response"],
                            "verdict": verdict_dict,
                            "judge_failed": judge_failed, "oom": False,
                            "ts": datetime.now(timezone.utc).isoformat(),
                        }
                        with open(results_path, "a") as f:
                            f.write(json.dumps(row) + "\n")
                    n_replays_done += 1
                    if n_replays_done % 5 == 0:
                        elapsed = time.time() - t_start
                        rate = n_replays_done / max(elapsed, 1)
                        remaining = n_replays_planned - n_replays_done
                        eta = remaining / max(rate, 1e-3) / 60
                        print(f"  [{n_replays_done}/{n_replays_planned}]  "
                              f"{rate:.2f} replay/s  eta {eta:.1f} min")

    print(f"\nFinished shard. {n_replays_done} replays done.")
    judge.unload()
    unload_model(vic_model, vic_tok)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay-pt", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--victim-model",
                   default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--probe-layer", type=int, default=16)
    p.add_argument("--steer-layer", type=int, default=None)
    p.add_argument("--alphas", nargs="+", type=float,
                   default=[-8.0, -4.0, -2.0, 0.0, 2.0, 4.0, 8.0])
    p.add_argument("--max-tokens", type=int, default=300)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--records-start", type=int, default=0)
    p.add_argument("--records-end", type=int, default=-1,
                   help="exclusive; -1 = use full dataset")
    p.add_argument("--direction-method", type=str, default="lr",
                   choices=["lr", "meandiff"],
                   help="how to fit the steering direction: "
                        "'lr' = logistic regression coefficient; "
                        "'meandiff' = mean(comply) - mean(refuse) (Arditi-style)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.steer_layer is None:
        args.steer_layer = args.probe_layer
    run_sweep(args)


if __name__ == "__main__":
    main()
