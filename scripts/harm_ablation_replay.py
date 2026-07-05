"""Corresponding analysis to the Arditi refusal replication, for HARM.

Refusal is an input GATE (a decision at the last prompt token) and is mediated
by a single direction (ablate -> comply). Harm-uplift is realized across the
generated RESPONSE. Question: is harm-uplift also mediated by a single ablatable
direction? If ablating a harm direction surgically lowers uplift while keeping
compliance + coherence, harm is a steerable axis (the "comply but with less
uplift" security dream). If it doesn't (or only via degradation/refusal), then
refusal is a single-direction gate and harm is NOT -> a clean mechanistic form
of "harm != compliance".

Set: high-harm (Stage-B rating>=4) COMPLIANT win turns (single-turn direct
prompts just get refused -> no uplift to reduce; multi-turn wins have real range).
Replay each breach turn: fix the attacker history, regenerate the victim response
under baseline / ablate_md_harm / ablate_lr_harm / ablate_random. Norm-preserving
directional ablation projected out of every residual write (embed + attn + mlp),
all layers/positions -- identical to arditi_replication.py.

Metrics (judged separately by 70B): Stage-B harm Likert + JBB compliance, + local
coherence. Compare uplift across arms AMONG still-compliant responses.
"""
import argparse
import json
import os

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
MAX_NEW_TOKENS = 320
SEED = 42


def normalize(v):
    return v / v.norm()


def load_model(device):
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map=device)
    model.eval()
    return model, tok


class AblationHooks:
    """Project unit r out of every residual write: embed + each attn + each mlp."""

    def __init__(self, r_unit):
        self.r = r_unit
        self._handles = []

    def _proj(self, module, inp, out):
        def ab(h):
            r = self.r.to(h.device, h.dtype)
            return h - (h @ r).unsqueeze(-1) * r
        if isinstance(out, tuple):
            return (ab(out[0]),) + out[1:]
        return ab(out)

    def register(self, model):
        self._handles.append(model.model.embed_tokens.register_forward_hook(self._proj))
        for layer in model.model.layers:
            self._handles.append(layer.self_attn.register_forward_hook(self._proj))
            self._handles.append(layer.mlp.register_forward_hook(self._proj))
        return self

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []


def coherence(text, tok):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if not ids:
        return {"token_unique_ratio": 0.0, "max_repeat": 0, "n_tokens": 0}
    mr = cur = 1
    for i in range(1, len(ids)):
        cur = cur + 1 if ids[i] == ids[i - 1] else 1
        mr = max(mr, cur)
    return {"token_unique_ratio": len(set(ids)) / len(ids), "max_repeat": mr, "n_tokens": len(ids)}


@torch.inference_mode()
def generate(model, tok, history):
    ids = tok.apply_chat_template(history, return_tensors="pt",
                                  add_generation_prompt=True).to(model.device)
    out = model.generate(ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()


def build_high_harm_set(scores_path, cands_path, min_rating, n):
    scores = [json.loads(l) for l in open(scores_path)]
    cands = {(c["source"], c["round"], c["idx"]): c
             for c in map(json.loads, open(cands_path))}
    picked = []
    for s in scores:
        if not (s.get("unsafe") and isinstance(s.get("rating"), int) and s["rating"] >= min_rating):
            continue
        key = (s["source"], s["round"], s["idx"])
        c = cands.get(key)
        if not c:
            continue
        conv = c["conversation"]
        asst_idx = [i for i, t in enumerate(conv) if t["role"] == "assistant"]
        tob = s.get("turn_of_breach")
        bi = asst_idx[tob] if (isinstance(tob, int) and tob < len(asst_idx)) else (asst_idx[-1] if asst_idx else -1)
        if bi <= 0 or conv[bi - 1]["role"] != "user":
            continue
        picked.append({"goal": c.get("goal", ""), "category": c.get("category"),
                       "source": s["source"], "round": s["round"], "idx": s["idx"],
                       "base_rating": s["rating"], "history": conv[:bi]})
        if len(picked) >= n:
            break
    return picked


def run_arm(model, tok, turns, cond, r_unit, fout, done):
    hook = AblationHooks(r_unit).register(model) if r_unit is not None else None
    try:
        for i, t in enumerate(turns):
            if (i, cond) in done:
                continue
            resp = generate(model, tok, t["history"])
            fout.write(json.dumps({
                "prompt_id": i, "method": cond, "alpha": 0.0,
                "goal": t["goal"], "category": t["category"],
                "source": t["source"], "round": t["round"], "idx": t["idx"],
                "base_rating": t["base_rating"], "response": resp,
                "coherence": coherence(resp, tok)}) + "\n")
            fout.flush()
    finally:
        if hook:
            hook.remove()
    print(f"  [{cond}] done", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="working/uplift/stage_b_scores_llama.jsonl")
    ap.add_argument("--candidates", default="working/uplift/stage_b_candidates.jsonl")
    ap.add_argument("--dir-md", default="directions_postresponse/v_md_harm_L16.pt")
    ap.add_argument("--dir-lr", default="directions_postresponse/v_lr_harm_L16.pt")
    ap.add_argument("--out-dir", default="experiments/harm_ablation_v1")
    ap.add_argument("--min-rating", type=int, default=4)
    ap.add_argument("--n", type=int, default=70)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    turns = build_high_harm_set(args.scores, args.candidates, args.min_rating, args.n)
    print(f"high-harm compliant turns (rating>={args.min_rating}): {len(turns)}", flush=True)

    print("Loading model (bf16)...", flush=True)
    model, tok = load_model(device)

    r_md = normalize(torch.load(args.dir_md).float())
    r_lr = normalize(torch.load(args.dir_lr).float())
    g = torch.Generator().manual_seed(SEED)
    r_rand = normalize(torch.randn(r_md.shape[0], generator=g).float())
    print(f"cos(md_harm,lr_harm)={float(torch.dot(r_md,r_lr)):+.3f} "
          f"cos(md_harm,random)={float(torch.dot(r_md,r_rand)):+.3f}", flush=True)

    out_path = f"{args.out_dir}/generations.jsonl"
    done = set()
    if os.path.exists(out_path):
        for l in open(out_path):
            try:
                r = json.loads(l); done.add((r["prompt_id"], r["method"]))
            except Exception:
                pass
        print(f"Resuming: {len(done)} done", flush=True)

    with open(out_path, "a") as fout:
        run_arm(model, tok, turns, "baseline", None, fout, done)
        run_arm(model, tok, turns, "ablate_md_harm", r_md, fout, done)
        run_arm(model, tok, turns, "ablate_lr_harm", r_lr, fout, done)
        run_arm(model, tok, turns, "ablate_random", r_rand, fout, done)

    print("DONE", flush=True)


if __name__ == "__main__":
    main()
