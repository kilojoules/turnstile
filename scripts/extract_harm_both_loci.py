"""Clean Fig-5 reproduction: WINS-ONLY (compliant turns; unsafe=True) harm probe at
BOTH loci from one pipeline. Isolates HARM readability from compliance (including losses
lets the input probe cheat via comply/refuse — that's why the all-turns run gave 0.89).
Expect: pre-response (input) ~mid-0.7s matching Fig 5; output-locus ~0.94.
"""
import json, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"; HS=17
dev="cuda" if torch.cuda.is_available() else "cpu"
tok=AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token is None: tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map=dev).eval()

@torch.inference_mode()
def both_reps(history, response):
    pre=tok.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
    pre_ids=tok(pre, return_tensors="pt").input_ids.to(dev)
    full_ids=tok(pre+response, return_tensors="pt").input_ids.to(dev)
    hs=model(full_ids, output_hidden_states=True, use_cache=False).hidden_states[HS][0].float().cpu().numpy()
    p=pre_ids.shape[1]
    return hs[p-1], (hs[p:].mean(0) if hs.shape[0]>p else hs[-1])

cands={(c["source"],c["round"],c["idx"]):c for c in map(json.loads, open("working/uplift/stage_b_candidates.jsonl"))}
Xin,Xout,rl,rq,gh=[],[],[],[],[]
nwin=nloss=0
for s in map(json.loads, open("working/uplift/stage_b_scores_llama.jsonl")):
    if not s.get("unsafe"): nloss+=1; continue      # WINS ONLY (compliant turns)
    nwin+=1
    lrr,qr=s.get("rating"),s.get("qwen_rating")
    if not isinstance(lrr,int) or not isinstance(qr,int): continue
    c=cands.get((s["source"],s["round"],s["idx"]))
    if not c: continue
    conv=c["conversation"]; ai=[i for i,t in enumerate(conv) if t["role"]=="assistant"]
    tob=s.get("turn_of_breach"); bi=ai[tob] if (isinstance(tob,int) and tob<len(ai)) else (ai[-1] if ai else -1)
    if bi<=0 or conv[bi-1]["role"]!="user": continue
    inp,out=both_reps(conv[:bi], conv[bi]["content"])
    Xin.append(inp); Xout.append(out); rl.append(lrr); rq.append(qr); gh.append(hash(str(c.get("goal","")))%(10**8))
Xin,Xout,rl,rq,gh=np.stack(Xin),np.stack(Xout),np.array(rl),np.array(rq),np.array(gh)
print(f"WINS-ONLY corpus: {nwin} wins ({nloss} losses excluded); usable={len(rl)}  harm≥4={(rl>=4).sum()} ≤2={(rl<=2).sum()}", flush=True)

def auc(X,y):
    m=(y>=4)|(y<=2); yb=(y[m]>=4).astype(int); Xm,gm=X[m],gh[m]; a=[]
    for tr,te in GroupKFold(5).split(Xm,yb,gm):
        c=LogisticRegression(C=1.0,class_weight="balanced",max_iter=2000).fit(Xm[tr],yb[tr])
        a.append(roc_auc_score(yb[te],c.decision_function(Xm[te])))
    return float(np.mean(a)),float(np.std(a))
res={}
for locus,X in [("INPUT (pre-response, Fig-5)",Xin),("OUTPUT (steering locus)",Xout)]:
    al=auc(X,rl); aq=auc(X,rq)
    print(f"HARM · {locus:<28} · Llama AUC={al[0]:.3f}±{al[1]:.3f}  Qwen AUC={aq[0]:.3f}±{aq[1]:.3f}", flush=True)
    res[locus]={"llama":al[0],"qwen":aq[0]}
json.dump({"wins_only":res,"n_wins":len(rl)}, open("harm_both_loci_auc.json","w"))
print("DONE", flush=True)
