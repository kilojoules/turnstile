"""Confirm the locus split from OUR pipeline: extract INPUT-locus harm reps (last user
token before the breach response, hidden_states[17]) — the Fig-5 pre-response locus we
discarded in build_directions — and compute GroupKFold AUC. Expect mid-0.7s (matching
Fig 5), vs 0.94 at the output locus. Also saves harm_dm_in.pt (input-locus harm dir).
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
def input_rep(history):  # last user token, pre-response (Fig-5 locus)
    ids=tok.apply_chat_template(history, return_tensors="pt", add_generation_prompt=True).to(dev)
    return model(ids, output_hidden_states=True, use_cache=False).hidden_states[HS][0,-1].float().cpu().numpy()

cands={(c["source"],c["round"],c["idx"]):c for c in map(json.loads, open("working/uplift/stage_b_candidates.jsonl"))}
X,rl,rq,gh=[],[],[],[]
for s in map(json.loads, open("working/uplift/stage_b_scores_llama.jsonl")):
    lrr,qr=s.get("rating"),s.get("qwen_rating")
    if not isinstance(lrr,int) or not isinstance(qr,int): continue
    c=cands.get((s["source"],s["round"],s["idx"]))
    if not c: continue
    conv=c["conversation"]; ai=[i for i,t in enumerate(conv) if t["role"]=="assistant"]
    tob=s.get("turn_of_breach"); bi=ai[tob] if (isinstance(tob,int) and tob<len(ai)) else (ai[-1] if ai else -1)
    if bi<=0 or conv[bi-1]["role"]!="user": continue
    X.append(input_rep(conv[:bi])); rl.append(lrr); rq.append(qr); gh.append(hash(str(c.get("goal","")))%(10**8))
X,rl,rq,gh=np.stack(X),np.array(rl),np.array(rq),np.array(gh)
print(f"input-locus harm reps: n={len(rl)}", flush=True)

def auc(y):
    m=(y>=4)|(y<=2); yb=(y[m]>=4).astype(int); Xm,gm=X[m],gh[m]; a=[]
    for tr,te in GroupKFold(5).split(Xm,yb,gm):
        c=LogisticRegression(C=1.0,class_weight="balanced",max_iter=2000).fit(Xm[tr],yb[tr])
        a.append(roc_auc_score(yb[te],c.decision_function(Xm[te])))
    return np.mean(a),np.std(a)
al=auc(rl); aq=auc(rq)
print(f"HARM · INPUT locus (pre-response, Fig-5) · Llama AUC = {al[0]:.3f} ± {al[1]:.3f}", flush=True)
print(f"HARM · INPUT locus (pre-response, Fig-5) · Qwen  AUC = {aq[0]:.3f} ± {aq[1]:.3f}", flush=True)
# save input-locus harm direction
m=(rl>=4)|(rl<=2); y=(rl[m]>=4)
w=torch.tensor(X[m][y].mean(0)-X[m][~y].mean(0)); torch.save((w/w.norm()).float(), "harm_dm_in.pt")
json.dump({"harm_input_llama":al[0],"harm_input_qwen":aq[0],"n":len(rl)}, open("input_harm_auc.json","w"))
print("DONE", flush=True)
