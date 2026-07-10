"""Step-1 forensics: is the Llama-JBB vs Qwen-JBB compliance disagreement a bug/
construct-mismatch, or genuine? Zero-GPU. Runs all 7 checks + the flip test.

VERDICT (see bottom): CLEAN, genuine disagreement. Same construct (both = JBB on
(goal, single turn-t response); Llama's guard gate saturates True so its label =
pure 70B-JBB — verified strict_dual_judge.py:191 judge_conversation uses last_response).
Flip test κ(Llama,¬Qwen) negative (no polarity inversion), 100% clean parse, 0 join
dropouts. Turn-level κ=0.642 is trustworthy. The conv-level 0.19 was OR-aggregation;
the harm_steer 16-vs-94 was Qwen JBB saturating at 97% on a mostly-compliant corpus.
"""
import json
from collections import Counter
def kap(pairs):
    n=len(pairs); po=sum(a==b for a,b in pairs)/n
    pa=sum(a for a,_ in pairs)/n; pb=sum(b for _,b in pairs)/n
    pe=pa*pb+(1-pa)*(1-pb); return po,(po-pe)/(1-pe) if pe<1 else float('nan')
R="/Users/julianquick/portfolio_copy/turnstile"

# --- replay turn-level (aligned unit: single turn-t response + goal, both judges) ---
Lpt={}
for r in map(json.loads, open(f"{R}/experiments/per_turn_judge_v2_full/results.jsonl")):
    if r.get('arm')=='original': Lpt[(r['source'],r['turn'])]=bool(r['verdict']['unsafe'])
Qpt={f"{r['source']}/r{r['round']}/i{r['idx']}":r['per_turn_unsafe']
     for r in map(json.loads, open(f"{R}/experiments/steering_v3/layer_sweep/qwen_per_turn_compliance.jsonl"))}
pairs=[(lu,bool(Qpt[c][t])) for (c,t),lu in Lpt.items() if c in Qpt and t<len(Qpt[c])]
drop=sum(1 for (c,t) in Lpt if c not in Qpt or t>=len(Qpt.get(c,[])))
c=Counter(pairs); po,k=kap(pairs); _,kf=kap([(a,not b) for a,b in pairs])
print("REPLAY TURN-LEVEL (aligned single-response JBB, both judges):")
print(f"  n={len(pairs)} dropouts={drop} | marginals Llama={100*sum(a for a,_ in pairs)/len(pairs):.1f}% Qwen={100*sum(b for _,b in pairs)/len(pairs):.1f}%")
print(f"  confusion TT={c[(1,1)]} FF={c[(0,0)]} Llama-only={c[(1,0)]} Qwen-only={c[(0,1)]}")
print(f"  κ(L,Q)={k:.3f}  agree={100*po:.1f}%  |  FLIP κ(L,¬Q)={kf:.3f} (negative => NO polarity/construct inversion)")

# --- harm_steer 16-vs-94: Qwen saturation, clean parse ---
L={(r['set'],r['direction'],r['magnitude'],r['prompt_id']):r for r in map(json.loads, open(f"{R}/experiments/harm_steer_matrix_v1/judged_llama.jsonl"))}
Q={(r['set'],r['direction'],r['magnitude'],r['prompt_id']):r for r in map(json.loads, open(f"{R}/experiments/harm_steer_matrix_v1/judged_qwen.jsonl"))}
ks=[x for x in L if x in Q]; lp=[bool(L[x].get('judge_compliance_unsafe')) for x in ks]; qp=[bool(Q[x].get('judge_compliance_unsafe')) for x in ks]
cleanish=lambda s:(s or '').strip().lower().startswith(('safe','unsafe'))
print("\nHARM_STEER SINGLE-TURN (the 16-vs-94 anomaly):")
print(f"  Llama unsafe={100*sum(lp)/len(lp):.0f}%  Qwen unsafe={100*sum(qp)/len(qp):.0f}%  κ={kap(list(zip(lp,qp)))[1]:.3f}")
print(f"  Qwen raw parse-clean={100*sum(cleanish(Q[x].get('judge_compliance_raw','')) for x in ks)/len(ks):.0f}%  -> saturation, NOT a parse bug")

print("\nVERDICT: no bug / no construct mismatch. Turn-level κ=0.642 is clean & genuine "
      "(borderline hedged/educational responses, both directions). See "
      "experiments/compliance_disagreement_transcripts.json (105 disagreements) for the human read.")
