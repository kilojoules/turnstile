"""Build the full direction inventory at L16 (hidden_states[17] = layer-16 output,
the exact locus the steering hook adds to). Re-extracts EVERY rep fresh so all
directions live in one consistent space; caches unit directions to directions/ and
the raw reps+labels to directions/reps.npz for local Phase-0 cosines, probe AUCs, SIR.

Inventory:
  refusal_dm      input  μ(harmful prompts) − μ(harmless prompts)        [defines REF_NORM]
  comp_probe      input  LR(complied vs refused)                         whitened compliance
  comp_dm_out     output μ(complied turns) − μ(refused turns)            locus-control
  comp_probe_out  output LR(complied vs refused)
  harm_dm_llama   output μ(harm≥4) − μ(harm≤2)   Llama labels            original lever
  harm_dm_qwen    output   same, Qwen labels
  harm_probe_llama output LR(harm≥4 vs ≤2)  Llama labels                 whitened harm
  harm_probe_qwen  output   same, Qwen labels
  length_dm       output μ(top-quartile len) − μ(bottom-quartile len)    confound
  random_1..5     random unit vectors
"""
import json, os
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.linear_model import LogisticRegression

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
HS = 17            # hidden_states index for layer-16 OUTPUT (== steering-hook locus)
SEED = 42
OUT = "directions"; os.makedirs(OUT, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"

def normalize(v): return v / v.norm()
def dm(X, y):  # diff-in-means, y boolean/0-1
    y = np.asarray(y); return X[y == 1].mean(0) - X[y == 0].mean(0)
def lr(X, y):
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, solver="lbfgs")
    clf.fit(X, y); return clf.coef_[0].astype(np.float32)

def main():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map=dev).eval()

    @torch.inference_mode()
    def prompt_last(prompt):
        ids = tok.apply_chat_template([{"role": "user", "content": prompt}], return_tensors="pt", add_generation_prompt=True).to(dev)
        return model(ids, output_hidden_states=True, use_cache=False).hidden_states[HS][0, -1].float().cpu().numpy()

    @torch.inference_mode()
    def turn_reps(history, response):  # returns (input last-user-token, output response-mean)
        pre = tok.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
        pre_ids = tok(pre, return_tensors="pt").input_ids.to(dev)
        full_ids = tok(pre + response, return_tensors="pt").input_ids.to(dev)
        hs = model(full_ids, output_hidden_states=True, use_cache=False).hidden_states[HS][0].float().cpu().numpy()
        p = pre_ids.shape[1]
        return hs[p-1], (hs[p:].mean(0) if hs.shape[0] > p else hs[-1])

    dirs, meta = {}, {}

    # ---- refusal_dm (input locus) : harmful goals vs alpaca, n=50 each (reproduces REF_NORM≈7.56)
    goals = [g["goal"] for g in json.load(open("working/uplift/goals.json"))][:50]
    alpaca = [x["instruction"] for x in json.load(open("data/alpaca_eval.json"))][:50]
    mu_h = np.stack([prompt_last(p) for p in goals]).mean(0)
    mu_b = np.stack([prompt_last(p) for p in alpaca]).mean(0)
    w = torch.tensor(mu_h - mu_b); REF_NORM = float(w.norm())
    dirs["refusal_dm"] = normalize(w); meta["REF_NORM"] = REF_NORM
    print(f"refusal_dm REF_NORM={REF_NORM:.3f}", flush=True)

    # ---- compliance corpus (replay_v2_full.pt): per-turn complied(unsafe=1)/refused(0)
    recs = torch.load("experiments/clamp_obliterated_v2/replay_v2_full.pt", weights_only=False)
    Xci, Xco, yc, gc = [], [], [], []
    for gi, r in enumerate(recs):
        conv = r.get("original_conversation"); labs = r.get("original_per_turn_labels")
        if not conv or not labs: continue
        ai = [i for i, t in enumerate(conv) if t["role"] == "assistant"]
        lab_by = {l["turn"]: l for l in labs if l.get("unsafe") is not None}
        for t, aidx in enumerate(ai):
            if t not in lab_by or aidx == 0 or conv[aidx-1]["role"] != "user": continue
            inp, out = turn_reps(conv[:aidx], conv[aidx]["content"])
            Xci.append(inp); Xco.append(out); yc.append(int(bool(lab_by[t]["unsafe"]))); gc.append(gi)
    Xci, Xco, yc, gc = np.stack(Xci), np.stack(Xco), np.array(yc), np.array(gc)
    print(f"compliance turns: n={len(yc)}  complied={yc.sum()} refused={(yc==0).sum()}  convs={len(set(gc))}", flush=True)
    dirs["comp_probe"] = normalize(torch.tensor(lr(Xci, yc)))
    dirs["comp_dm_out"] = normalize(torch.tensor(dm(Xco, yc)))
    dirs["comp_probe_out"] = normalize(torch.tensor(lr(Xco, yc)))

    # ---- harm corpus (stage_b): mean-pooled response reps + llama/qwen ratings + length
    cands = {(c["source"], c["round"], c["idx"]): c for c in map(json.loads, open("working/uplift/stage_b_candidates.jsonl"))}
    Xh, rl, rq, gh, ln = [], [], [], [], []
    for s in map(json.loads, open("working/uplift/stage_b_scores_llama.jsonl")):
        lrr, qr = s.get("rating"), s.get("qwen_rating")
        if not isinstance(lrr, int) or not isinstance(qr, int): continue
        c = cands.get((s["source"], s["round"], s["idx"]))
        if not c: continue
        conv = c["conversation"]; ai = [i for i, t in enumerate(conv) if t["role"] == "assistant"]
        tob = s.get("turn_of_breach"); bi = ai[tob] if (isinstance(tob, int) and tob < len(ai)) else (ai[-1] if ai else -1)
        if bi <= 0 or conv[bi-1]["role"] != "user": continue
        resp = conv[bi]["content"]
        _, out = turn_reps(conv[:bi], resp)
        Xh.append(out); rl.append(lrr); rq.append(qr); gh.append(hash(str(c.get("goal", ""))) % (10**8))
        ln.append(len(tok(resp, add_special_tokens=False)["input_ids"]))
    Xh, rl, rq, gh, ln = np.stack(Xh), np.array(rl), np.array(rq), np.array(gh), np.array(ln)
    print(f"harm wins: n={len(rl)}  llama≥4={(rl>=4).sum()}/≤2={(rl<=2).sum()}  qwen≥4={(rq>=4).sum()}/≤2={(rq<=2).sum()}", flush=True)

    def harm_dirs(rating, tag):
        m = (rating >= 4) | (rating <= 2); y = (rating[m] >= 4).astype(int)
        dirs[f"harm_dm_{tag}"] = normalize(torch.tensor(dm(Xh[m], y)))
        dirs[f"harm_probe_{tag}"] = normalize(torch.tensor(lr(Xh[m], y)))
    harm_dirs(rl, "llama"); harm_dirs(rq, "qwen")

    q1, q3 = np.percentile(ln, 25), np.percentile(ln, 75)
    ylen = np.where(ln >= q3, 1, np.where(ln <= q1, 0, -1))
    mlen = ylen >= 0
    dirs["length_dm"] = normalize(torch.tensor(dm(Xh[mlen], ylen[mlen])))
    print(f"length_dm: q1={q1:.0f} q3={q3:.0f} tokens", flush=True)

    # ---- random controls
    g = torch.Generator().manual_seed(SEED)
    for k in range(1, 6):
        dirs[f"random_{k}"] = normalize(torch.randn(4096, generator=g))

    for name, v in dirs.items():
        torch.save(v.float(), f"{OUT}/{name}.pt")
    meta["dirs"] = list(dirs.keys()); meta["HS_index"] = HS
    json.dump(meta, open(f"{OUT}/meta.json", "w"), indent=2)
    np.savez(f"{OUT}/reps.npz", Xci=Xci, Xco=Xco, yc=yc, gc=gc,
             Xh=Xh, rl=rl, rq=rq, gh=gh, ln=ln)
    print(f"saved {len(dirs)} directions + reps.npz", flush=True)

if __name__ == "__main__":
    main()
