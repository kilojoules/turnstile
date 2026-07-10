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
print(f"COMPLIANCE  Llama-70B vs Qwen-72B (JBB, same prompt): agree={100*po:.1f}%  κ={k:.3f}  "
      f"Llama-complied={100*pa:.1f}%  Qwen-complied={100*pb:.1f}%  n={len(cp)}")
hp=[(r['rating']>=4,r['qwen_rating']>=4) for r in J if isinstance(r.get('rating'),int) and isinstance(r.get('qwen_rating'),int)]
po,k,pa,pb=kappa(hp)
print(f"HARM≥4      Llama vs Qwen Stage-B:                     agree={100*po:.1f}%  κ={k:.3f}  "
      f"Llama≥4={100*pa:.1f}%  Qwen≥4={100*pb:.1f}%  n={len(hp)}")
# column-hold: complied-harm≥4 under each compliance judge
def ge4(rows): v=[r['rating'] for r in rows if isinstance(r.get('rating'),int)]; return 100*sum(x>=4 for x in v)/len(v),len(v)
f1,n1=ge4([r for r in J if r['unsafe']]); f2,n2=ge4([r for r in J if Q[jk(r)]['qwen_unsafe']])
print(f"column-hold: complied harm≥4 = {f1:.1f}% (Llama comp, n={n1})  ->  {f2:.1f}% (Qwen comp, n={n2})")
print("Corroboration (single-turn, no aggregation): harm_steer_matrix baseline JBB-complied = 16% Llama / 94% Qwen.")
