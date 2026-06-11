"""Shared probe primitives for the intent-obliteration analyses.

All three analyses (intent_obliteration, intent_text_baseline,
intent_topic_matched) call into this module so they use identical CV, AUC,
permutation-test and seed-handling code.
"""

from __future__ import annotations

import json
from typing import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def auc_cv_dense(X: np.ndarray, y: np.ndarray, seed: int, n_splits: int = 5) -> dict:
    """5-fold stratified CV AUC for dense features (hidden states).

    Returns: {"mean": float, "std": float, "per_fold": [float, ...]}.
    """
    X = np.asarray(X)
    y = np.asarray(y).astype(int)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
        clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])[:, 1]
        folds.append(float(roc_auc_score(y[te], p)))
    return {
        "mean": float(np.mean(folds)),
        "std": float(np.std(folds)),
        "per_fold": folds,
    }


def auc_cv_text(texts, y: np.ndarray, seed: int, n_splits: int = 5) -> dict:
    """5-fold stratified CV AUC for raw text via TF-IDF + logreg."""
    texts = np.asarray(texts, dtype=object)
    y = np.asarray(y).astype(int)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for tr, te in skf.split(texts, y):
        vec = TfidfVectorizer(
            ngram_range=(1, 2), min_df=2, max_features=20000,
            sublinear_tf=True, lowercase=True,
        )
        X_tr = vec.fit_transform(texts[tr])
        X_te = vec.transform(texts[te])
        clf = LogisticRegression(max_iter=2000, C=1.0, solver="liblinear")
        clf.fit(X_tr, y[tr])
        p = clf.predict_proba(X_te)[:, 1]
        folds.append(float(roc_auc_score(y[te], p)))
    return {
        "mean": float(np.mean(folds)),
        "std": float(np.std(folds)),
        "per_fold": folds,
    }


def seed_sweep_dense(X: np.ndarray, y: np.ndarray, seeds: Sequence[int]) -> dict:
    """Run dense-feature CV at each seed; return per-seed and aggregate stats."""
    per_seed = []
    means = []
    for s in seeds:
        r = auc_cv_dense(X, y, seed=s)
        per_seed.append({"seed": int(s), **r})
        means.append(r["mean"])
    return {
        "n_examples": int(len(y)),
        "n_pos": int(y.sum()),
        "n_neg": int(len(y) - y.sum()),
        "n_features": int(X.shape[1]),
        "seeds": [int(s) for s in seeds],
        "auc_seed_mean": float(np.mean(means)),
        "auc_seed_std": float(np.std(means)),
        "per_seed": per_seed,
    }


def seed_sweep_text(texts, y: np.ndarray, seeds: Sequence[int]) -> dict:
    per_seed = []
    means = []
    for s in seeds:
        r = auc_cv_text(texts, y, seed=s)
        per_seed.append({"seed": int(s), **r})
        means.append(r["mean"])
    return {
        "n_examples": int(len(y)),
        "n_pos": int(y.sum()),
        "n_neg": int(len(y) - y.sum()),
        "seeds": [int(s) for s in seeds],
        "auc_seed_mean": float(np.mean(means)),
        "auc_seed_std": float(np.std(means)),
        "per_seed": per_seed,
    }


def permutation_test_dense(X: np.ndarray, y: np.ndarray,
                           n_perms: int = 5, seed: int = 0) -> dict:
    """Validity check: shuffle labels and re-fit. AUC should be ~0.5."""
    rng = np.random.default_rng(seed)
    aucs = []
    for i in range(n_perms):
        y_perm = y.copy()
        rng.shuffle(y_perm)
        r = auc_cv_dense(X, y_perm, seed=int(rng.integers(0, 2**31 - 1)))
        aucs.append(r["mean"])
    return {
        "n_perms": n_perms,
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "per_perm": aucs,
    }


def permutation_test_text(texts, y: np.ndarray,
                          n_perms: int = 5, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    aucs = []
    for i in range(n_perms):
        y_perm = y.copy()
        rng.shuffle(y_perm)
        r = auc_cv_text(texts, y_perm, seed=int(rng.integers(0, 2**31 - 1)))
        aucs.append(r["mean"])
    return {
        "n_perms": n_perms,
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "per_perm": aucs,
    }


def balanced_subsample_seed_sweep_dense(
    X_pos: np.ndarray, X_neg: np.ndarray,
    n_per_class: int, seeds: Sequence[int],
) -> dict:
    """Balanced subsample: at each seed, draw n_per_class from each side, run CV."""
    per_seed = []
    means = []
    for s in seeds:
        rng = np.random.default_rng(s)
        n = min(n_per_class, len(X_pos), len(X_neg))
        ai = rng.choice(len(X_pos), n, replace=False)
        bi = rng.choice(len(X_neg), n, replace=False)
        X = np.concatenate([X_pos[ai], X_neg[bi]], axis=0)
        y = np.concatenate([np.ones(n), np.zeros(n)])
        r = auc_cv_dense(X, y, seed=int(s))
        per_seed.append({"seed": int(s), "n_per_class": int(n), **r})
        means.append(r["mean"])
    return {
        "n_per_class": int(n),
        "seeds": [int(s) for s in seeds],
        "auc_seed_mean": float(np.mean(means)),
        "auc_seed_std": float(np.std(means)),
        "per_seed": per_seed,
    }


def balanced_subsample_seed_sweep_text(
    texts_pos: list, texts_neg: list,
    n_per_class: int, seeds: Sequence[int],
) -> dict:
    per_seed = []
    means = []
    for s in seeds:
        rng = np.random.default_rng(s)
        n = min(n_per_class, len(texts_pos), len(texts_neg))
        ai = rng.choice(len(texts_pos), n, replace=False)
        bi = rng.choice(len(texts_neg), n, replace=False)
        texts = ([texts_pos[i] for i in ai]
                 + [texts_neg[i] for i in bi])
        y = np.concatenate([np.ones(n), np.zeros(n)])
        r = auc_cv_text(texts, y, seed=int(s))
        per_seed.append({"seed": int(s), "n_per_class": int(n), **r})
        means.append(r["mean"])
    return {
        "n_per_class": int(n),
        "seeds": [int(s) for s in seeds],
        "auc_seed_mean": float(np.mean(means)),
        "auc_seed_std": float(np.std(means)),
        "per_seed": per_seed,
    }


def manifest_dense(X: np.ndarray, y: np.ndarray) -> dict:
    """Sanity-check stats about the activation/text matrix being probed."""
    return {
        "n_examples": int(len(y)),
        "n_pos": int(y.sum()),
        "n_neg": int(len(y) - y.sum()),
        "n_features": int(X.shape[1]),
        "pos_norm_mean": float(np.linalg.norm(X[y == 1], axis=1).mean()),
        "pos_norm_std": float(np.linalg.norm(X[y == 1], axis=1).std()),
        "neg_norm_mean": float(np.linalg.norm(X[y == 0], axis=1).mean()),
        "neg_norm_std": float(np.linalg.norm(X[y == 0], axis=1).std()),
    }


def write_json(obj, path: str) -> None:
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
