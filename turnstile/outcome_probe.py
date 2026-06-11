"""Outcome prediction probe: can the model's internal state predict jailbreak success?

The previous probe (0.97 AUC) distinguished adversarial vs. benign conversations —
nearly tautological since the inputs differ structurally. This experiment asks the
harder question: among adversarial conversations only, can turn-0 activations predict
which attacks will eventually succeed?

Two modes:
  --local     Use pre-extracted hidden states at layer 16 (no GPU needed)
  --gpu       Extract hidden states at all layers on GPU (layer sweep)

Usage:
  # Phase A1: local, pre-extracted hidden states at L16
  python -m turnstile.outcome_probe \
      --hs-dir experiments/stealth_s42/hidden_states \
      --rounds-dir experiments/stealth_s42/rounds \
      --output experiments/outcome_probe_v1

  # Phase A2: GPU layer sweep (run on Vast.ai)
  python -m turnstile.outcome_probe \
      --rounds-dir experiments/stealth_s42/rounds \
      --gpu --layer-sweep \
      --output experiments/outcome_probe_v1
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score


def _natural_sort_key(fname):
    """Sort round_0.jsonl, round_1.jsonl, ..., round_10.jsonl naturally."""
    base = fname.replace(".jsonl", "").replace(".pt", "")
    parts = base.split("_")
    return int(parts[-1]) if parts[-1].isdigit() else 0


# ---------------------------------------------------------------------------
# Phase A1: Load pre-extracted hidden states (no GPU needed)
# ---------------------------------------------------------------------------

def load_preextracted(hs_dir, rounds_dir):
    """Load pre-extracted hidden states and merge with category info from JSONL.

    Returns per-turn data dict: turn_idx -> {hs, labels, categories, turns_of_breach}
    """
    # Load hidden states
    hs_files = sorted(
        [f for f in os.listdir(hs_dir) if f.startswith("round_") and f.endswith(".pt")],
        key=_natural_sort_key,
    )

    # Load round JSONL for category info
    round_files = sorted(
        [f for f in os.listdir(rounds_dir) if f.startswith("round_") and f.endswith(".jsonl")],
        key=_natural_sort_key,
    )

    turn_data = defaultdict(lambda: {"hs": [], "labels": [], "categories": [],
                                      "turns_of_breach": [], "goals": []})

    for hs_file in hs_files:
        round_num = int(hs_file.replace("round_", "").replace(".pt", ""))
        hs_path = os.path.join(hs_dir, hs_file)
        data = torch.load(hs_path, weights_only=False)

        # Match with JSONL for categories
        jsonl_file = f"round_{round_num}.jsonl"
        categories = []
        goals = []
        if jsonl_file in round_files:
            with open(os.path.join(rounds_dir, jsonl_file)) as f:
                for line in f:
                    conv = json.loads(line)
                    categories.append(conv.get("category", "unknown"))
                    goals.append(conv.get("goal", ""))
        else:
            categories = ["unknown"] * len(data["labels"])
            goals = [""] * len(data["labels"])

        # Unpack per-turn hidden states
        for i, (hs_tensor, label) in enumerate(zip(data["hidden_states"], data["labels"])):
            # hs_tensor: (num_turns, d_model)
            tob = data["turns_of_breach"][i]
            cat = categories[i] if i < len(categories) else "unknown"
            goal = goals[i] if i < len(goals) else ""

            for turn_idx in range(hs_tensor.shape[0]):
                turn_data[turn_idx]["hs"].append(hs_tensor[turn_idx])
                turn_data[turn_idx]["labels"].append(int(label))
                turn_data[turn_idx]["categories"].append(cat)
                turn_data[turn_idx]["turns_of_breach"].append(tob)
                turn_data[turn_idx]["goals"].append(goal)

    return turn_data


# ---------------------------------------------------------------------------
# Phase A2: Extract hidden states on GPU (for layer sweep)
# ---------------------------------------------------------------------------

def extract_perturn_hidden_states_gpu(model, tokenizer, rounds_dir, layer_idx,
                                      max_convs=500):
    """Extract hidden states at every turn for a specific layer. GPU required.

    Returns per-turn dict: turn_idx -> {hs: list, labels: list, categories: list}
    """
    from turnstile.model_utils import extract_hidden_states

    round_files = sorted(
        [f for f in os.listdir(rounds_dir) if f.startswith("round_") and f.endswith(".jsonl")],
        key=_natural_sort_key,
    )

    turn_data = defaultdict(lambda: {"hs": [], "labels": [], "categories": []})
    count = 0

    for rf in round_files:
        with open(os.path.join(rounds_dir, rf)) as f:
            for line in f:
                conv = json.loads(line)
                messages = conv.get("conversation", [])
                if not messages:
                    continue

                label = 1 if conv.get("unsafe") else 0
                category = conv.get("category", "unknown")

                # Build prompts at each turn (before each victim response)
                accumulated = []
                turn_idx = 0
                for msg in messages:
                    accumulated.append(msg)
                    if msg["role"] == "user" and turn_idx < 5:
                        prompt = tokenizer.apply_chat_template(
                            accumulated, tokenize=False, add_generation_prompt=True
                        )
                        try:
                            h = extract_hidden_states(
                                model, tokenizer, [prompt],
                                layer_idx=layer_idx, max_length=512,
                            )
                            turn_data[turn_idx]["hs"].append(h.squeeze(0))
                            turn_data[turn_idx]["labels"].append(label)
                            turn_data[turn_idx]["categories"].append(category)
                        except RuntimeError:
                            break
                        turn_idx += 1

                count += 1
                if count >= max_convs:
                    break
                if count % 100 == 0:
                    print(f"    Extracted {count}/{max_convs}")

        if count >= max_convs:
            break

    return turn_data


# ---------------------------------------------------------------------------
# Probe training
# ---------------------------------------------------------------------------

def train_probe(X, y, n_splits=5):
    """Train logistic regression with stratified CV. Returns metrics + fitted clf."""
    if len(np.unique(y)) < 2:
        return {"auc": 0.5, "auc_std": 0.0, "n_pos": 0, "n_neg": len(y)}, None

    min_class = min(int(y.sum()), int(len(y) - y.sum()))
    if min_class < n_splits:
        n_splits = max(2, min_class)

    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, solver="lbfgs")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    aucs = cross_val_score(clf, X, y, cv=skf, scoring="roc_auc")

    clf.fit(X, y)

    return {
        "auc": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "n_pos": int(y.sum()),
        "n_neg": int(len(y) - y.sum()),
    }, clf


def category_cv(X, y, categories):
    """Leave-one-category-out CV. Tests generalization across harm categories."""
    unique_cats = list(set(categories))
    if len(unique_cats) < 3:
        return None

    cat_array = np.array(categories)
    aucs = []
    per_cat = {}

    for held_out in sorted(unique_cats):
        mask = cat_array == held_out
        if mask.sum() < 2 or (~mask).sum() < 2:
            continue
        if len(np.unique(y[mask])) < 2:
            continue

        clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, solver="lbfgs")
        clf.fit(X[~mask], y[~mask])
        y_prob = clf.predict_proba(X[mask])[:, 1]
        auc = roc_auc_score(y[mask], y_prob)
        aucs.append(auc)
        per_cat[held_out] = {"auc": auc, "n": int(mask.sum()),
                             "n_pos": int(y[mask].sum())}

    if not aucs:
        return None

    return {
        "auc": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "n_categories": len(aucs),
        "per_category": per_cat,
    }


# ---------------------------------------------------------------------------
# Direction comparison
# ---------------------------------------------------------------------------

def compare_directions(outcome_dir, output_dir):
    """Compute cosine similarities with existing directions."""
    comparisons = {}

    existing = {
        "arditi_L31": "experiments/clamp_v1/refusal_direction_L31.pt",
        "probe_L31": "experiments/clamp_v2_probe/probe_direction_L31.pt",
        "probe_L16": "experiments/clamp_v2_probe/probe_direction_L16.pt",
    }

    for name, path in existing.items():
        if os.path.exists(path):
            d = torch.load(path, weights_only=True)
            if d.shape == outcome_dir.shape:
                cos = float(torch.dot(outcome_dir, d))
                comparisons[name] = cos

    return comparisons


# ---------------------------------------------------------------------------
# Main: local mode (Phase A1)
# ---------------------------------------------------------------------------

def run_local(args):
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    print("=== PHASE A1: Outcome Prediction (Local, Pre-extracted L16) ===\n")

    # Load data
    turn_data = load_preextracted(args.hs_dir, args.rounds_dir)
    n_turns = len(turn_data)
    n_total = len(turn_data[0]["labels"])
    n_pos = sum(turn_data[0]["labels"])
    n_neg = n_total - n_pos
    print(f"Loaded {n_total} conversations ({n_pos} wins, {n_neg} losses), {n_turns} turns\n")

    # Turn-of-breach distribution
    tob_counts = defaultdict(int)
    for tob in turn_data[0]["turns_of_breach"]:
        tob_counts[tob] = tob_counts.get(tob, 0) + 1
    print("Turn-of-breach distribution:")
    for t in sorted(tob_counts.keys(), key=lambda x: (x is None, x)):
        label = str(t) if t is not None else "None (no breach)"
        print(f"  Turn {label}: {tob_counts[t]}")
    print()

    all_results = {}

    # === Experiment 1: Per-turn outcome prediction ===
    print("=" * 60)
    print("EXPERIMENT 1: Per-turn outcome prediction (layer 16)")
    print("=" * 60)

    per_turn = {}
    outcome_directions = {}

    for turn_idx in sorted(turn_data.keys()):
        td = turn_data[turn_idx]
        X = torch.stack(td["hs"]).numpy()
        y = np.array(td["labels"])

        metrics, clf = train_probe(X, y)
        per_turn[turn_idx] = metrics

        if clf is not None:
            direction = torch.from_numpy(clf.coef_[0]).float()
            direction = direction / direction.norm()
            outcome_directions[turn_idx] = direction
            torch.save(direction, os.path.join(
                output_dir, f"outcome_direction_L16_turn{turn_idx}.pt"
            ))

        print(f"  Turn {turn_idx}: AUC = {metrics['auc']:.4f} +/- {metrics['auc_std']:.4f} "
              f"({metrics['n_pos']} wins, {metrics['n_neg']} losses)")

    all_results["per_turn_L16"] = per_turn

    # === Experiment 2: Category-matched CV (turn 0) ===
    print(f"\n{'=' * 60}")
    print("EXPERIMENT 2: Category-matched leave-one-out CV (turn 0, L16)")
    print("=" * 60)

    td0 = turn_data[0]
    X0 = torch.stack(td0["hs"]).numpy()
    y0 = np.array(td0["labels"])
    cats = td0["categories"]

    cat_result = category_cv(X0, y0, cats)
    if cat_result:
        print(f"  Overall AUC: {cat_result['auc']:.4f} +/- {cat_result['auc_std']:.4f} "
              f"({cat_result['n_categories']} categories)")
        print(f"\n  Per-category (held-out):")
        for cat_name, cr in sorted(cat_result["per_category"].items(),
                                    key=lambda x: x[1]["auc"], reverse=True):
            print(f"    {cat_name:>30s}: AUC={cr['auc']:.4f} "
                  f"(n={cr['n']}, {cr['n_pos']} wins)")
        all_results["category_cv_turn0"] = cat_result
    else:
        print("  Not enough data for category CV")

    # === Experiment 3: Direction comparisons ===
    print(f"\n{'=' * 60}")
    print("EXPERIMENT 3: Direction comparisons")
    print("=" * 60)

    comparisons = {}

    if 0 in outcome_directions:
        # vs existing directions
        ext = compare_directions(outcome_directions[0], output_dir)
        for name, cos in sorted(ext.items()):
            print(f"  cos(outcome_L16_t0, {name}) = {cos:.4f}")
            comparisons[f"outcome_L16_t0 vs {name}"] = cos

        # across turns
        for t in range(1, 5):
            if t in outcome_directions:
                cos = float(torch.dot(outcome_directions[0], outcome_directions[t]))
                comparisons[f"outcome_t0 vs outcome_t{t}"] = cos
                print(f"  cos(outcome_t0, outcome_t{t}) = {cos:.4f}")

    all_results["direction_comparisons"] = comparisons

    # === Summary ===
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)

    print(f"\n  Per-turn outcome AUC (layer 16):")
    print(f"  {'Turn':>4s}  {'AUC':>8s}  {'Std':>6s}")
    for t in sorted(per_turn.keys()):
        r = per_turn[t]
        print(f"  {t:>4d}  {r['auc']:>8.4f}  {r['auc_std']:>6.4f}")

    # Save
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to {output_dir}/results.json")
    print("PHASE A1 COMPLETE")


# ---------------------------------------------------------------------------
# Main: GPU mode (Phase A2 — layer sweep)
# ---------------------------------------------------------------------------

def run_gpu(args):
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    print("=== PHASE A2: Full Layer x Turn Sweep for Outcome Prediction (GPU) ===\n")

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    from turnstile.model_utils import load_model, unload_model

    model, tokenizer = load_model(args.victim_model)

    layers = [0, 4, 8, 12, 16, 20, 24, 28, 31]
    # Full matrix: layer x turn -> AUC
    full_matrix = {}  # key: "L{layer}_T{turn}" -> metrics

    for layer_idx in layers:
        print(f"\n  Layer {layer_idx}...")
        turn_data = extract_perturn_hidden_states_gpu(
            model, tokenizer, args.rounds_dir,
            layer_idx=layer_idx, max_convs=args.max_convs,
        )

        for turn_idx in sorted(turn_data.keys()):
            td = turn_data[turn_idx]
            if not td["hs"]:
                continue

            X = torch.stack(td["hs"]).numpy()
            y = np.array(td["labels"])
            metrics, clf = train_probe(X, y)

            key = f"L{layer_idx}_T{turn_idx}"
            full_matrix[key] = metrics
            full_matrix[key]["layer"] = layer_idx
            full_matrix[key]["turn"] = turn_idx

            if clf is not None:
                direction = torch.from_numpy(clf.coef_[0]).float()
                direction = direction / direction.norm()
                torch.save(direction, os.path.join(
                    output_dir, f"outcome_direction_L{layer_idx}_turn{turn_idx}.pt"
                ))

            print(f"    Turn {turn_idx}: AUC = {metrics['auc']:.4f} +/- {metrics['auc_std']:.4f}")

    # Load A1 results if available and merge
    results_path = os.path.join(output_dir, "results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            all_results = json.load(f)
    else:
        all_results = {}

    all_results["full_layer_turn_matrix"] = full_matrix

    # Print full matrix
    print(f"\n{'=' * 60}")
    print("FULL LAYER x TURN OUTCOME AUC MATRIX")
    print("=" * 60)
    print(f"  {'Layer':>5s}", end="")
    for t in range(5):
        print(f"  Turn {t:d}", end="")
    print()
    for l in layers:
        print(f"  L{l:>3d} ", end="")
        for t in range(5):
            key = f"L{l}_T{t}"
            if key in full_matrix:
                auc = full_matrix[key]["auc"]
                print(f"  {auc:.4f}", end="")
            else:
                print(f"     -  ", end="")
        print()

    # Find peak
    if full_matrix:
        peak = max(full_matrix.items(), key=lambda x: x[1]["auc"])
        print(f"\n  Peak: {peak[0]} AUC = {peak[1]['auc']:.4f}")

    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to {results_path}")
    print("PHASE A2 COMPLETE")

    unload_model(model, tokenizer)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Outcome prediction probe")
    parser.add_argument("--hs-dir", type=str, default=None,
                        help="Pre-extracted hidden states dir (for local mode)")
    parser.add_argument("--rounds-dir", type=str, required=True,
                        help="Directory with round_N.jsonl files")
    parser.add_argument("--victim-model", type=str,
                        default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--max-convs", type=int, default=500,
                        help="Max conversations for GPU extraction")
    parser.add_argument("--gpu", action="store_true",
                        help="Run GPU layer sweep (Phase A2)")
    parser.add_argument("--output", type=str,
                        default="experiments/outcome_probe_v1")
    args = parser.parse_args()

    if args.gpu:
        run_gpu(args)
    elif args.hs_dir:
        run_local(args)
    else:
        parser.error("Specify --hs-dir for local mode or --gpu for GPU mode")


if __name__ == "__main__":
    main()
