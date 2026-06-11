"""MLP probe at L16/T1 + L12/T1 on pooled 9400 convs.

Tests whether linear compliance AUC 0.74 is probe-capacity limited.
If MLP matches linear → ceiling likely near-Bayes.
If MLP >> linear → ceiling is a linearity artifact, paper must pivot.
"""
import gc
import json
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

POOLED = "experiments/pooled_hs"


def load(layer, turn):
    hs, y = [], []
    for src in sorted(os.listdir(POOLED)):
        sd = f"{POOLED}/{src}"
        if not os.path.isdir(sd): continue
        for f in sorted(os.listdir(sd)):
            if not f.endswith(".pt"): continue
            d = torch.load(f"{sd}/{f}", weights_only=False)
            arr = d["hidden_states_by_layer"][layer].numpy()
            for i in range(arr.shape[0]):
                hs.append(arr[i, turn])
                y.append(int(bool(d["labels"][i])))
            del d, arr
    gc.collect()
    return np.stack(hs), np.array(y)


class MLP(nn.Module):
    def __init__(self, d_in=4096, d_hidden=256, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp(X_tr, y_tr, X_val, y_val, d_hidden=256, lr=1e-3,
              wd=1e-3, epochs=40, batch=256, pos_weight=None, device="cpu"):
    Xt = torch.from_numpy(X_tr).float().to(device)
    yt = torch.from_numpy(y_tr).float().to(device)
    Xv = torch.from_numpy(X_val).float().to(device)
    yv_np = y_val
    model = MLP(Xt.shape[1], d_hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    bce = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, device=device) if pos_weight else None)
    n = len(Xt)
    best = 0.5
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch):
            idx = perm[i:i+batch]
            opt.zero_grad()
            logits = model(Xt[idx])
            loss = bce(logits, yt[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vp = torch.sigmoid(model(Xv)).cpu().numpy()
        auc = roc_auc_score(yv_np, vp)
        if auc > best: best = auc
    return best


def cv_auc(X, y, model_type="linear", seed=42, **mlp_kw):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    aucs = []
    pos_weight = (y == 0).sum() / (y == 1).sum()
    for tr, te in skf.split(X, y):
        if model_type == "linear":
            clf = LogisticRegression(C=1.0, class_weight="balanced",
                                      max_iter=2000, solver="lbfgs")
            clf.fit(X[tr], y[tr])
            p = clf.predict_proba(X[te])[:, 1]
            aucs.append(roc_auc_score(y[te], p))
        else:
            auc = train_mlp(X[tr], y[tr], X[te], y[te],
                            pos_weight=pos_weight, **mlp_kw)
            aucs.append(auc)
    return float(np.mean(aucs)), float(np.std(aucs))


def main():
    print("Loading L16 T1...", flush=True)
    X, y = load(16, 1)
    print(f"  shape {X.shape}, wins {int(y.sum())}/{len(y)}", flush=True)

    print("\n=== LINEAR baseline ===", flush=True)
    lin_mean, lin_std = cv_auc(X, y, "linear")
    print(f"Linear L16/T1: {lin_mean:.4f} ± {lin_std:.4f}", flush=True)

    out = {"L16_T1_linear": {"auc": lin_mean, "std": lin_std}}

    for cfg in [
        {"d_hidden": 128, "wd": 1e-3, "epochs": 30},
        {"d_hidden": 256, "wd": 1e-3, "epochs": 30},
        {"d_hidden": 512, "wd": 1e-3, "epochs": 30},
        {"d_hidden": 256, "wd": 1e-2, "epochs": 30},  # heavier reg
    ]:
        print(f"\n=== MLP {cfg} ===", flush=True)
        mean, std = cv_auc(X, y, "mlp", **cfg)
        print(f"MLP L16/T1: {mean:.4f} ± {std:.4f}  "
              f"(delta linear = {mean-lin_mean:+.4f})", flush=True)
        out[f"L16_T1_mlp_h{cfg['d_hidden']}_wd{cfg['wd']}"] = {
            "auc": mean, "std": std, "cfg": cfg,
        }

    # Also L12/T1 (peak cell)
    print("\nLoading L12 T1...", flush=True)
    X12, y12 = load(12, 1)
    lin12 = cv_auc(X12, y12, "linear")
    print(f"Linear L12/T1: {lin12[0]:.4f} ± {lin12[1]:.4f}", flush=True)
    mlp12 = cv_auc(X12, y12, "mlp", d_hidden=256, wd=1e-3, epochs=30)
    print(f"MLP L12/T1 (h=256): {mlp12[0]:.4f} ± {mlp12[1]:.4f}  "
          f"(delta = {mlp12[0]-lin12[0]:+.4f})", flush=True)
    out["L12_T1_linear"] = {"auc": lin12[0], "std": lin12[1]}
    out["L12_T1_mlp_h256"] = {"auc": mlp12[0], "std": mlp12[1]}

    os.makedirs("experiments/outcome_probe_v1", exist_ok=True)
    with open("experiments/outcome_probe_v1/mlp_probe.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nWrote experiments/outcome_probe_v1/mlp_probe.json", flush=True)


if __name__ == "__main__":
    main()
