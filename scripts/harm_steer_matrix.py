"""Final harm-steering matrix: {Qwen-harm, Llama-harm, random} × {low, high uplift}.

Fits TWO post-response harm difference-in-means directions from the SAME win corpus,
differing ONLY in the judge whose Likert labels define high/low uplift:
  - harm_llama : Llama-70B rating >=4 (pos) vs <=2 (neg)     [stage_b_scores_llama 'rating']
  - harm_qwen  : Qwen-72B  rating >=4 (pos) vs <=2 (neg)     [same file, 'qwen_rating']
Then steers each direction (+ a random control) by adding it at L16 across a magnitude
grid, on two eval sets (low-uplift rating<=3 and high-uplift rating>=4 compliant replies,
disjoint from the fit set). Saves responses so a dual judge pass (Llama + Qwen) can score
harm + compliance. Answers "can we steer harm up?" robustly to both the direction-defining
judge and the outcome-scoring judge.
"""
import argparse, json, os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
LAYER = 16
MAX_NEW_TOKENS = 320
SEED = 42
MAGNITUDES = [0.0, 2.8, 5.5, 11.1]   # ‖added vector‖ (raw harm norm ~1.4, residual norm ~9.7)

def normalize(v): return v / v.norm()
def coherence(text, tok):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if not ids: return {"token_unique_ratio": 0.0, "max_repeat": 0, "n_tokens": 0}
    mr = cur = 1
    for i in range(1, len(ids)):
        cur = cur + 1 if ids[i] == ids[i-1] else 1; mr = max(mr, cur)
    return {"token_unique_ratio": len(set(ids))/len(ids), "max_repeat": mr, "n_tokens": len(ids)}

class AddHook:
    def __init__(self, vec): self.v = vec; self._h = None
    def _add(self, m, i, o):
        if isinstance(o, tuple): return (o[0]+self.v.to(o[0].device, o[0].dtype),)+o[1:]
        return o+self.v.to(o.device, o.dtype)
    def register(self, model): self._h = model.model.layers[LAYER].register_forward_hook(self._add); return self
    def remove(self):
        if self._h: self._h.remove(); self._h = None

@torch.inference_mode()
def resp_rep(model, tok, history, response):
    pre = tok.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
    pre_ids = tok(pre, return_tensors="pt").input_ids.to(model.device)
    full_ids = tok(pre + response, return_tensors="pt").input_ids.to(model.device)
    hs = model(full_ids, output_hidden_states=True, use_cache=False).hidden_states[LAYER+1][0]
    return hs[pre_ids.shape[1]:].float().mean(0).cpu()

@torch.inference_mode()
def gen(model, tok, history):
    ids = tok.apply_chat_template(history, return_tensors="pt", add_generation_prompt=True).to(model.device)
    o = model.generate(ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(o[0][ids.shape[1]:], skip_special_tokens=True).strip()

def load_corpus(scores_path, cands_path):
    """One row per win with BOTH llama+qwen ratings, history, response, key."""
    cands = {(c["source"], c["round"], c["idx"]): c for c in map(json.loads, open(cands_path))}
    out = []
    for s in map(json.loads, open(scores_path)):
        lr = s.get("rating"); qr = s.get("qwen_rating")
        if not isinstance(lr, int) or not isinstance(qr, int): continue
        c = cands.get((s["source"], s["round"], s["idx"]))
        if not c: continue
        conv = c["conversation"]; ai = [i for i, t in enumerate(conv) if t["role"] == "assistant"]
        tob = s.get("turn_of_breach"); bi = ai[tob] if (isinstance(tob, int) and tob < len(ai)) else (ai[-1] if ai else -1)
        if bi <= 0 or conv[bi-1]["role"] != "user": continue
        out.append({"llama_rating": lr, "qwen_rating": qr, "unsafe": s.get("unsafe"),
                    "goal": c.get("goal", ""), "category": c.get("category"),
                    "history": conv[:bi], "response": conv[bi]["content"],
                    "key": (s["source"], s["round"], s["idx"])})
    return out

def fit_dir(model, tok, pos, neg):
    H = torch.stack([resp_rep(model, tok, d["history"], d["response"]) for d in pos]).mean(0)
    L = torch.stack([resp_rep(model, tok, d["history"], d["response"]) for d in neg]).mean(0)
    w = H - L
    return normalize(w), float(w.norm())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="working/uplift/stage_b_scores_llama.jsonl")
    ap.add_argument("--candidates", default="working/uplift/stage_b_candidates.jsonl")
    ap.add_argument("--out-dir", default="experiments/harm_steer_matrix_v1")
    ap.add_argument("--n-fit", type=int, default=70)
    ap.add_argument("--n-eval", type=int, default=32)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = load_corpus(args.scores, args.candidates)
    llama_hi = [d for d in data if d["llama_rating"] >= 4][:args.n_fit]
    llama_lo = [d for d in data if d["llama_rating"] <= 2][:args.n_fit]
    qwen_hi = [d for d in data if d["qwen_rating"] >= 4][:args.n_fit]
    qwen_lo = [d for d in data if d["qwen_rating"] <= 2][:args.n_fit]
    fit_keys = {d["key"] for d in llama_hi+llama_lo+qwen_hi+qwen_lo}
    low_eval = [d for d in data if d.get("unsafe") and d["llama_rating"] <= 3 and d["key"] not in fit_keys][:args.n_eval]
    high_eval = [d for d in data if d.get("unsafe") and d["llama_rating"] >= 4 and d["key"] not in fit_keys][:args.n_eval]
    print(f"fit: llama {len(llama_hi)}/{len(llama_lo)}  qwen {len(qwen_hi)}/{len(qwen_lo)} | "
          f"eval low={len(low_eval)} high={len(high_eval)} (disjoint)", flush=True)

    print("Loading model (bf16)...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map=device).eval()

    print("Fitting harm directions (Llama-labelled and Qwen-labelled)...", flush=True)
    v_llama, n_llama = fit_dir(model, tok, llama_hi, llama_lo)
    v_qwen, n_qwen = fit_dir(model, tok, qwen_hi, qwen_lo)
    g = torch.Generator().manual_seed(SEED); v_rand = normalize(torch.randn(v_llama.shape[0], generator=g).float())
    cos = float(torch.dot(v_llama, v_qwen))
    print(f"  raw_norm llama={n_llama:.3f} qwen={n_qwen:.3f}  cos(harm_llama,harm_qwen)={cos:+.3f}", flush=True)
    json.dump({"raw_norm_llama": n_llama, "raw_norm_qwen": n_qwen, "cos_llama_qwen": cos,
               "magnitudes": MAGNITUDES, "n_low": len(low_eval), "n_high": len(high_eval)},
              open(f"{args.out_dir}/meta.json", "w"), indent=2)

    dirs = {"harm_llama": v_llama, "harm_qwen": v_qwen, "random": v_rand}
    out = f"{args.out_dir}/sweep.jsonl"; done = set()
    if os.path.exists(out):
        for l in open(out):
            try: r = json.loads(l); done.add((r["set"], r["direction"], r["magnitude"], r["prompt_id"]))
            except Exception: pass
    fout = open(out, "a")
    def emit(setn, dirn, mag, i, d, resp):
        fout.write(json.dumps({"set": setn, "direction": dirn, "magnitude": mag, "prompt_id": i,
            "goal": d["goal"], "category": d["category"], "llama_rating": d["llama_rating"],
            "qwen_rating": d["qwen_rating"], "response": resp, "coherence": coherence(resp, tok),
            "method": f"{setn}_{dirn}_m{mag:g}"}) + "\n"); fout.flush()

    for setn, ev in [("low", low_eval), ("high", high_eval)]:
        for i, d in enumerate(ev):
            if (setn, "baseline", 0.0, i) in done: continue
            emit(setn, "baseline", 0.0, i, d, gen(model, tok, d["history"]))
        print(f"  [{setn}/baseline] done", flush=True)
        for dirn, unit in dirs.items():
            for mag in MAGNITUDES:
                if mag == 0.0: continue
                hook = AddHook(unit*mag).register(model)
                try:
                    for i, d in enumerate(ev):
                        if (setn, dirn, mag, i) in done: continue
                        emit(setn, dirn, mag, i, d, gen(model, tok, d["history"]))
                finally: hook.remove()
            print(f"  [{setn}/{dirn}] done", flush=True)
    fout.close(); print("DONE", flush=True)

if __name__ == "__main__":
    main()
