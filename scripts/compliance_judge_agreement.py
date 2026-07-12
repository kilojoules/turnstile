"""Compliance-judge label stability check (zero-GPU). Cross-tabs the two JBB
compliance instruments (Llama-3.1-70B `unsafe` vs Qwen2.5-72B `qwen_unsafe`, same
JBB prompt) on the 600 stage-B rows, against the two Stage-B harm judges. Finding:
compliance LABELS are judge-fragile (κ≈0.19) while harm LABELS are judge-stable
(κ≈0.77) — the inverse of the naive asymmetry, and distinct from Fig-5 decode-stability.
"""
import json
ROOT="/Users/julianquick/portfolio_copy/turnstile"
J=[json.loads(l) for l in open(f"{ROOT}/working/uplift/stage_b_scores_llama.jsonl")]
Q={r.get('conv_id') or f"{r['source']}/r{r['round']}/i{r['idx']}":r
   for r in map(json.loads, open(f"{ROOT}/experiments/steering_v3/layer_sweep/qwen_per_turn_compliance.jsonl"))}
def jk(r): return r.get('conv_id') or f"{r['source']}/r{r['round']}/i{r['idx']}"

def kappa(pairs):
    n=len(pairs); po=sum(a==b for a,b in pairs)/n
    pa=sum(a for a,_ in pairs)/n; pb=sum(b for _,b in pairs)/n
    pe=pa*pb+(1-pa)*(1-pb); return po,(po-pe)/(1-pe),pa,pb

cp=[(bool(r['unsafe']),bool(Q[jk(r)]['qwen_unsafe'])) for r in J if jk(r) in Q]
po,k,pa,pb=kappa(cp)
print(f"COMPLIANCE conv-level (any-turn agg; ARTIFACT-PRONE): agree={100*po:.1f}%  κ={k:.3f}  "
      f"Llama={100*pa:.1f}%  Qwen={100*pb:.1f}%  n={len(cp)}")

# CLEAN turn-level compliance: per-turn Llama JBB (per_turn_judge, original arm) vs Qwen per-turn
Lpt={}
for r in map(json.loads, open(f"{ROOT}/experiments/per_turn_judge_v2_full/results.jsonl")):
    if r.get('arm')=='original': Lpt[(r['source'],r['turn'])]=bool(r['verdict']['unsafe'])
Qpt={f"{r['source']}/r{r['round']}/i{r['idx']}":r['per_turn_unsafe']
     for r in map(json.loads, open(f"{ROOT}/experiments/steering_v3/layer_sweep/qwen_per_turn_compliance.jsonl"))}
tp=[(lu,bool(Qpt[c][t])) for (c,t),lu in Lpt.items() if c in Qpt and t<len(Qpt[c])]
po,k,pa,pb=kappa(tp)
print(f"COMPLIANCE TURN-LEVEL (clean, no aggregation):        agree={100*po:.1f}%  κ={k:.3f}  "
      f"Llama={100*pa:.1f}%  Qwen={100*pb:.1f}%  n={len(tp)} turns/{len(set(c for c,_ in Lpt))} convs")
hp=[(r['rating']>=4,r['qwen_rating']>=4) for r in J if isinstance(r.get('rating'),int) and isinstance(r.get('qwen_rating'),int)]
po,k,pa,pb=kappa(hp)
print(f"HARM≥4      Llama vs Qwen Stage-B (binary, matched):   agree={100*po:.1f}%  κ={k:.3f}  "
      f"Llama≥4={100*pa:.1f}%  Qwen≥4={100*pb:.1f}%  n={len(hp)}")
# harm is ORDINAL 1-5: report weighted κ (quadratic = standard for Likert)
import numpy as np
from sklearn.metrics import cohen_kappa_score
a=np.array([r['rating'] for r in J if isinstance(r.get('rating'),int) and isinstance(r.get('qwen_rating'),int)])
b=np.array([r['qwen_rating'] for r in J if isinstance(r.get('rating'),int) and isinstance(r.get('qwen_rating'),int)])
print(f"HARM 1-5    ordinal κ:  unweighted(5-way exact)={cohen_kappa_score(a,b):.3f}  "
      f"linear-wt={cohen_kappa_score(a,b,weights='linear'):.3f}  "
      f"QUADRATIC-wt={cohen_kappa_score(a,b,weights='quadratic'):.3f} (standard)  exact={100*np.mean(a==b):.1f}% within1={100*np.mean(abs(a-b)<=1):.1f}%")
print("  NOTE: compliance is BINARY (weighted==unweighted). Contrast must be granularity-matched:")
print("        binary-vs-binary compliance 0.642 vs harm≥4 0.765; proper ordinal harm κ=0.811."
      " Only unweighted-5-way harm (0.544) vs binary compliance makes harm look less-agreed = artifact.")
# column-hold: complied-harm≥4 under each compliance judge
def ge4(rows): v=[r['rating'] for r in rows if isinstance(r.get('rating'),int)]; return 100*sum(x>=4 for x in v)/len(v),len(v)
f1,n1=ge4([r for r in J if r['unsafe']]); f2,n2=ge4([r for r in J if Q[jk(r)]['qwen_unsafe']])
print(f"column-hold: complied harm≥4 = {f1:.1f}% (Llama comp, n={n1})  ->  {f2:.1f}% (Qwen comp, n={n2})")
print("Corroboration (single-turn, no aggregation): harm_steer_matrix baseline JBB-complied = 16% Llama / 94% Qwen.")
