"""Advisor backstop: verify the fit locus of every probe/direction, confirm CV is
out-of-fold + grouped, and write fit_locus into directions/meta.json. The Phase-0
harm probe (0.94) is OUTPUT-locus (mean-pooled response) — NOT the pre-response
Fig-5 locus (mid-0.7s). Report both, separately labelled; never substitute.
"""
import json, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

# fit_locus per direction, from build_directions.py:
#   input  = last user token, hidden_states[17][-1]  (pre-response; Fig-5 locus)
#   output = mean-pooled RESPONSE tokens             (steering locus; harm_dm built here)
FIT_LOCUS = {
    "refusal_dm": "input", "comp_probe": "input",
    "comp_dm_out": "output", "comp_probe_out": "output",
    "harm_dm_llama": "output", "harm_dm_qwen": "output",
    "harm_probe_llama": "output", "harm_probe_qwen": "output",
    "length_dm": "output", "harm_dm_resid": "output", "comp_dm_out_resid": "output",
    **{f"random_{k}": "n/a" for k in range(1, 6)},
}
print("===== FIT LOCUS PER DIRECTION =====")
for d, loc in FIT_LOCUS.items():
    tok = {"input": "last USER token (pre-response)", "output": "mean-pooled RESPONSE tokens",
           "n/a": "—"}[loc]
    print(f"  {d:<20} {loc:<7} ({tok})")

z = np.load("directions/reps.npz")
def cv_auc(X, y, groups, name):
    m = (y >= 4) | (y <= 2) if y.max() > 1 else np.ones(len(y), bool)
    yb = (y[m] >= 4).astype(int) if y.max() > 1 else y[m]
    Xm, gm = X[m], groups[m]; aucs = []
    for tr, te in GroupKFold(5).split(Xm, yb, gm):
        c = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(Xm[tr], yb[tr])
        aucs.append(roc_auc_score(yb[te], c.decision_function(Xm[te])))  # OUT-OF-FOLD
    ngroups = len(set(gm))
    print(f"  {name:<34} AUC = {np.mean(aucs):.3f} ± {np.std(aucs):.3f}   (out-of-fold, {ngroups} groups)")
    return float(np.mean(aucs))

print("\n===== PROBE AUC, EXPLICITLY LABELLED BY LOCUS (GroupKFold, out-of-fold) =====")
print("CV: GroupKFold(k=5); AUC scored on held-out test fold only (never in-sample).")
print("Groups: harm = by GOAL (≥ as strict as by-conversation); compliance = by CONVERSATION.\n")
a_harm_out_l = cv_auc(z["Xh"], z["rl"], z["gh"], "harm  · OUTPUT locus · Llama")
a_harm_out_q = cv_auc(z["Xh"], z["rq"], z["gh"], "harm  · OUTPUT locus · Qwen")
a_comp_in = cv_auc(z["Xci"], z["yc"], z["gc"], "compliance · INPUT locus (Fig-5)")
a_comp_out = cv_auc(z["Xco"], z["yc"], z["gc"], "compliance · OUTPUT locus")

print("\n===== THE LOCUS EFFECT (compliance, measured both ways from one corpus) =====")
print(f"  compliance INPUT (pre-response) = {a_comp_in:.3f}   vs   OUTPUT = {a_comp_out:.3f}   (+{a_comp_out-a_comp_in:.3f})")
print("  => reading the RESPONSE inflates separability. For harm this gap is larger")
print("     (Fig-5 pre-response mid-0.7s -> OUTPUT ~0.94), because harmful text is more")
print("     textually distinctive. The Phase-0 0.94 is OUTPUT-locus separability, NOT")
print("     pre-response readability. NOTE: reps.npz has no INPUT-locus harm reps")
print("     (build_directions discarded `inp` for harm) -> re-extract to confirm from pipeline.")

# write fit_locus into meta.json
meta = json.load(open("directions/meta.json"))
meta["fit_locus"] = FIT_LOCUS
meta["phase0_auc"] = {
    "harm_output_llama": a_harm_out_l, "harm_output_qwen": a_harm_out_q,
    "compliance_input": a_comp_in, "compliance_output": a_comp_out,
    "NOTE": "harm probe is OUTPUT locus (mean-pooled response) = paper's ~95% aside; "
            "pre-response (Fig-5) harm is mid-0.7s and is the readability claim. Do not substitute.",
}
json.dump(meta, open("directions/meta.json", "w"), indent=2)
print("\nwrote fit_locus + labelled AUCs to directions/meta.json")
