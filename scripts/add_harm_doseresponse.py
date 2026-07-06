"""ADD-harm DOSE-RESPONSE: does harm rise with push strength, and at what coherence cost?

fig4 showed ADD-harm at one calibrated magnitude (+0.11). But the earlier additive
sweep found a larger directional harm signal (+0.40 'odd') at much bigger magnitude,
entangled with degradation. This resolves the tension: sweep the ADD magnitude from 0
into the degradation regime, for the harm direction AND a random direction (to subtract
the pure norm effect), and track BOTH Stage-B uplift and coherence.

Magnitudes are multiples of the harm diff-in-means raw norm (~1.4) so m=1 is the natural
'calibrated' dose (fig4's 1x) and large m pushes toward a full residual-stream norm
(~9.7 at L16), where any direction degrades the model.
"""
import argparse, json, os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
LAYER = 16
MAX_NEW_TOKENS = 320
SEED = 42
MULTS = [0.0, 1.0, 2.0, 4.0, 8.0]   # x raw diff-in-means norm (m=8 ≈ full residual norm)

def normalize(v): return v / v.norm()

def load_model(device):
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map=device).eval()
    return m, tok

class AddHook:
    def __init__(self, vec): self.v = vec; self._h = None
    def _add(self, mod, i, o):
        if isinstance(o, tuple): return (o[0] + self.v.to(o[0].device, o[0].dtype),) + o[1:]
        return o + self.v.to(o.device, o.dtype)
    def register(self, model): self._h = model.model.layers[LAYER].register_forward_hook(self._add); return self
    def remove(self):
        if self._h: self._h.remove(); self._h = None

def coherence(text, tok):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if not ids: return {"token_unique_ratio": 0.0, "max_repeat": 0, "n_tokens": 0}
    mr = cur = 1
    for i in range(1, len(ids)):
        cur = cur + 1 if ids[i] == ids[i-1] else 1; mr = max(mr, cur)
    return {"token_unique_ratio": len(set(ids))/len(ids), "max_repeat": mr, "n_tokens": len(ids)}

@torch.inference_mode()
def resp_rep(model, tok, history, response):
    pre = tok.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
    pre_ids = tok(pre, return_tensors="pt").input_ids.to(model.device)
    full_ids = tok(pre + response, return_tensors="pt").input_ids.to(model.device)
    hs = model(full_ids, output_hidden_states=True, use_cache=False).hidden_states[LAYER+1][0]
    return hs[pre_ids.shape[1]:].float().mean(0).cpu()

def load_scored(scores_path, cands_path):
    scores = [json.loads(l) for l in open(scores_path)]
    cands = {(c["source"], c["round"], c["idx"]): c for c in map(json.loads, open(cands_path))}
    out = []
    for s in scores:
        if not isinstance(s.get("rating"), int): continue
        c = cands.get((s["source"], s["round"], s["idx"]))
        if not c: continue
        conv = c["conversation"]; ai = [i for i, t in enumerate(conv) if t["role"] == "assistant"]
        tob = s.get("turn_of_breach"); bi = ai[tob] if (isinstance(tob, int) and tob < len(ai)) else (ai[-1] if ai else -1)
        if bi <= 0 or conv[bi-1]["role"] != "user": continue
        out.append({"rating": s["rating"], "unsafe": s.get("unsafe"), "goal": c.get("goal", ""),
                    "category": c.get("category"), "history": conv[:bi], "response": conv[bi]["content"],
                    "key": (s["source"], s["round"], s["idx"])})
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="working/uplift/stage_b_scores_llama.jsonl")
    ap.add_argument("--candidates", default="working/uplift/stage_b_candidates.jsonl")
    ap.add_argument("--out-dir", default="experiments/add_harm_doseresponse_v1")
    ap.add_argument("--n-fit", type=int, default=80)
    ap.add_argument("--n-eval", type=int, default=45)
    ap.add_argument("--eval-min-rating", type=int, default=1)
    ap.add_argument("--eval-max-rating", type=int, default=3)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = load_scored(args.scores, args.candidates)
    hi = [d for d in data if d["rating"] >= 4][:args.n_fit]
    lo = [d for d in data if d["rating"] <= 2][:args.n_fit]
    fit_keys = {d["key"] for d in hi + lo}   # eval must be DISJOINT from the fit set
    ev = [d for d in data if d.get("unsafe") and args.eval_min_rating <= d["rating"] <= args.eval_max_rating
          and d["key"] not in fit_keys][:args.n_eval]
    print(f"fit hi={len(hi)} lo={len(lo)} | eval={len(ev)} (rating {args.eval_min_rating}-{args.eval_max_rating}, disjoint from fit)", flush=True)

    print("Loading model...", flush=True)
    model, tok = load_model(device)
    print("Fitting raw harm diff-in-means...", flush=True)
    H = torch.stack([resp_rep(model, tok, d["history"], d["response"]) for d in hi]).mean(0)
    L = torch.stack([resp_rep(model, tok, d["history"], d["response"]) for d in lo]).mean(0)
    raw = H - L; raw_norm = float(raw.norm()); unit = normalize(raw)
    g = torch.Generator().manual_seed(SEED); rnd = normalize(torch.randn(raw.shape[0], generator=g).float())
    print(f"  raw_norm={raw_norm:.3f}", flush=True)
    json.dump({"raw_norm": raw_norm, "mults": MULTS}, open(f"{args.out_dir}/meta.json", "w"))

    out = f"{args.out_dir}/generations.jsonl"; done = set()
    if os.path.exists(out):
        for l in open(out):
            try: r = json.loads(l); done.add((r["prompt_id"], r["method"]))
            except Exception: pass
    fout = open(out, "a")
    arms = []
    for m in MULTS:
        arms.append((f"harm_m{m:g}", unit * (m * raw_norm)))
        if m > 0: arms.append((f"random_m{m:g}", rnd * (m * raw_norm)))
    for name, vec in arms:
        hook = AddHook(vec).register(model) if vec.norm() > 0 else None
        try:
            for i, d in enumerate(ev):
                if (i, name) in done: continue
                ids = tok.apply_chat_template(d["history"], return_tensors="pt", add_generation_prompt=True).to(model.device)
                with torch.inference_mode():
                    o = model.generate(ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, pad_token_id=tok.eos_token_id)
                resp = tok.decode(o[0][ids.shape[1]:], skip_special_tokens=True).strip()
                fout.write(json.dumps({"prompt_id": i, "method": name, "alpha": 0.0, "goal": d["goal"],
                                       "category": d["category"], "orig_rating": d["rating"],
                                       "response": resp, "coherence": coherence(resp, tok)}) + "\n")
                fout.flush()
        finally:
            if hook: hook.remove()
        print(f"  [{name}] done", flush=True)
    fout.close(); print("DONE", flush=True)

if __name__ == "__main__":
    main()
