"""Train SAEs at multiple latent dims, probe turn-0 compliance AUC on Network hacking.

For each n_features:
  1. Train SAE on pooled L16 turn-0 hidden states (all experiments, 9400 samples)
  2. Encode 400 Network hacking malicious turn-0 activations through SAE
  3. Train logistic regression probe on SAE features → 5-fold CV AUC
  4. Plot n_features vs AUC

Baseline = raw hidden states probe (no SAE).
"""
import gc
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, ".")
from turnstile.probe import SparseAutoencoder, normalize_activations

DEVICE = "cpu"  # MPS syncs break perf under Rosetta
print(f"Device: {DEVICE}", flush=True)

OUT_DIR = "experiments/sae_dim_sweep"
os.makedirs(OUT_DIR, exist_ok=True)

# Dims to sweep
DIMS = [256, 512, 1024, 2048, 4096, 8192, 16384]
TRAIN_STEPS = 2000
BATCH = 512
LR = 1e-3
L1_COEFF = 0.1  # lower: keep features alive


def load_sae_training_data():
    """All L16 turn-0 activations from pooled_hs as SAE training set."""
    print("Loading SAE training data (L16 T0 from pooled_hs)...")
    X = []
    pooled = "experiments/pooled_hs"
    for src in sorted(os.listdir(pooled)):
        sd = f"{pooled}/{src}"
        if not os.path.isdir(sd): continue
        for f in sorted(os.listdir(sd)):
            if not f.endswith(".pt"): continue
            d = torch.load(f"{sd}/{f}", weights_only=False)
            arr = d["hidden_states_by_layer"][16].numpy()  # (n, 5, 4096)
            for i in range(arr.shape[0]):
                X.append(arr[i, 0])
            del d, arr
    X = np.stack(X)
    print(f"  {X.shape}")
    return torch.from_numpy(X).float()


def load_nethack_probe_data():
    """400 Network hacking malicious L16 T0 + labels."""
    print("Loading Network hacking probe data...")
    import json as J
    Xs, ys = [], []
    d = torch.load("experiments/network_hacking_corpus/malicious/hs.pt",
                    weights_only=False)
    for h, y in zip(d["hidden_states"], d["labels"].tolist()):
        Xs.append(h[0].numpy())
        ys.append(int(bool(y)))
    fd = "experiments/nethack_focused/hidden_states"
    for f in sorted(os.listdir(fd)):
        if not f.endswith(".pt"): continue
        d = torch.load(f"{fd}/{f}", weights_only=False)
        for h, y in zip(d["hidden_states"], d["labels"].tolist()):
            Xs.append(h[0].numpy())
            ys.append(int(bool(y)))
    sources = ['authority_dpo','control_hard_s456','control_s42','frozen_v1',
               'incrementalism_dpo','reward_dpo','stealth_hard_s456',
               'stealth_jbb_v1','stealth_s42','urgency_dpo','urgency_v1']
    for src in sources:
        sd = f"experiments/pooled_hs/{src}"
        rd = f"experiments/{src}/rounds"
        if not (os.path.isdir(sd) and os.path.isdir(rd)): continue
        for fn in sorted(os.listdir(sd)):
            if not fn.endswith(".pt"): continue
            rn = int(fn.replace("round_","").replace(".pt",""))
            jpath = f"{rd}/round_{rn}.jsonl"
            if not os.path.exists(jpath): continue
            rows = [J.loads(l) for l in open(jpath)]
            data = torch.load(f"{sd}/{fn}", weights_only=False)
            arr = data["hidden_states_by_layer"][16].numpy()
            for i, r in enumerate(rows):
                if r.get("behavior") != "Network hacking": continue
                Xs.append(arr[i, 0])
                ys.append(int(bool(r.get("unsafe"))))
    X = np.stack(Xs)
    y = np.array(ys)
    print(f"  {X.shape}, wins={y.sum()}/{len(y)}")
    return torch.from_numpy(X).float(), y


def train_sae(X, n_features, steps=TRAIN_STEPS, batch=BATCH, lr=LR, l1=L1_COEFF):
    """Quick SAE training with normalized inputs."""
    d_model = X.shape[1]
    sae = SparseAutoencoder(d_model, n_features, l1_coeff=l1).to(DEVICE)
    # Nudge encoder bias so ReLU isn't dead at init
    with torch.no_grad():
        sae.encoder.bias.fill_(0.01)
    X_norm, scale = normalize_activations(X)
    X_norm = X_norm.to(DEVICE)

    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    sae.train()
    n = X_norm.shape[0]
    t0 = time.time()
    for step in range(steps):
        idx = torch.randint(0, n, (batch,), device=DEVICE)
        batch_x = X_norm[idx]
        loss, mse, l1v = sae.compute_loss(batch_x)
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % 500 == 0:
            print(f"    step {step+1}/{steps}  loss={loss.item():.3f} "
                  f"mse={mse:.3f} l1={l1v:.3f}", flush=True)
    print(f"    trained in {time.time()-t0:.1f}s", flush=True)
    sae.eval()
    return sae, float(scale)


def encode(sae, X, scale, device=DEVICE, batch=512):
    sae.eval()
    X_t = X.float() * scale
    feats = []
    with torch.no_grad():
        for i in range(0, len(X_t), batch):
            f = sae.encode(X_t[i:i+batch].to(device)).cpu().numpy()
            feats.append(f)
    return np.concatenate(feats).astype(np.float32)


def kfold_auc(X, y, k=5, seed=42):
    if len(np.unique(y))<2: return None
    k = min(k, int(y.sum()), int(len(y)-y.sum()))
    if k<2: return None
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    aucs = []
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(C=1.0, class_weight="balanced",
                                  max_iter=2000, solver="lbfgs")
        clf.fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
    return float(np.mean(aucs)), float(np.std(aucs))


def main():
    X_train = load_sae_training_data()
    X_probe, y_probe = load_nethack_probe_data()

    # Raw baseline
    print("\n=== Raw baseline (no SAE) ===")
    raw_auc = kfold_auc(X_probe.numpy(), y_probe)
    print(f"Raw L16 T0 compliance AUC: {raw_auc[0]:.4f} ± {raw_auc[1]:.4f}")

    results = {"raw": {"auc": raw_auc[0], "std": raw_auc[1]},
               "sae": {}}

    for d in DIMS:
        print(f"\n=== SAE n_features={d} ===")
        sae, scale = train_sae(X_train, d)
        feats = encode(sae, X_probe, scale)
        nz = (feats > 1e-6).sum(axis=1).mean()
        print(f"  encoded {feats.shape}, mean L0={nz:.1f}")
        r = kfold_auc(feats, y_probe)
        if r is None:
            print("  CV failed")
            continue
        results["sae"][d] = {"auc": r[0], "std": r[1], "mean_l0": float(nz)}
        print(f"  AUC = {r[0]:.4f} ± {r[1]:.4f}  delta vs raw = {r[0]-raw_auc[0]:+.4f}")
        del sae, feats
        gc.collect()

    with open(f"{OUT_DIR}/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {OUT_DIR}/results.json")

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dims = sorted(results["sae"].keys())
    aucs = [results["sae"][d]["auc"] for d in dims]
    stds = [results["sae"][d]["std"] for d in dims]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(dims, aucs, yerr=stds, marker="o", color="#e45a3b",
                capsize=4, linewidth=2, markersize=8, label="SAE features")
    ax.axhline(raw_auc[0], color="#264653", linestyle="--", linewidth=1.5,
               label=f"Raw hidden states ({raw_auc[0]:.3f})")
    ax.fill_between(dims, raw_auc[0]-raw_auc[1], raw_auc[0]+raw_auc[1],
                     color="#264653", alpha=0.1)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, label="chance")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("SAE latent dimension (n_features)")
    ax.set_ylabel("Turn-0 compliance prediction AUC (5-fold CV)")
    ax.set_title(f"Network Hacking: SAE dim × compliance AUC  "
                  f"(n={len(y_probe)} convs, {int(y_probe.sum())} wins)")
    ax.set_xticks(dims)
    ax.set_xticklabels([str(d) for d in dims])
    ax.legend(loc="lower right", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/sae_dim_vs_auc.png", dpi=150, bbox_inches="tight")
    plt.savefig(f"{OUT_DIR}/sae_dim_vs_auc.pdf", bbox_inches="tight")
    plt.close()
    print(f"Wrote {OUT_DIR}/sae_dim_vs_auc.png")


if __name__ == "__main__":
    main()
