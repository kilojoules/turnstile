"""Output-content positive control for the harm-ablation null.

The harm-ablation null (ablating the harm direction doesn't reduce uplift) is
only informative IF this ablation method can move output-content properties at
all. The only proven positive control so far (refusal) is an INPUT-GATE concept
fit on prompts. This script asks: can the identical norm-preserving projection
ablation remove an OUTPUT-CONTENT property fit on responses (exactly like harm)?

Two properties spanning the gate-ish -> distributed spectrum, both auto-measured
(no LLM judge):
  - LANGUAGE (English vs French): decided early, gate-ish. metric = langdetect.
  - VERBOSITY (terse vs verbose): distributed across the whole response, like
    harm. metric = response token count.

For each property, fit a diff-in-means direction on POST-RESPONSE reps (with vs
without the property), then:
  ADD arm  (sufficiency, Arditi Fig 3 analog): on 'without' prompts, add the raw
    direction -> does the property appear? vs norm-matched random-add.
  ABLATE arm (necessity, Arditi Fig 1 analog, = the test harm FAILED): on 'with'
    prompts, project the unit direction out of every residual write -> does the
    property disappear? vs random-ablate + baseline.

If ablation removes these output properties, the method CAN touch output content
and the harm null is informative. If not, the method is output-blind and the
harm null is uninterpretable for the mechanistic claim.
"""
import argparse
import json
import os

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
LAYER = 16
MAX_NEW_TOKENS = 200
SEED = 42

PROPS = {
    "french": {
        "sys_with": "Tu es un assistant utile. Réponds toujours entièrement en français, quelle que soit la langue de la question.",
        "sys_without": "You are a helpful assistant. Always respond in English.",
    },
    "verbose": {
        "sys_with": "You are a helpful assistant. Always respond in an extremely long, exhaustively detailed answer of many paragraphs.",
        "sys_without": "You are a helpful assistant. Always respond in exactly one short sentence.",
    },
}


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
    def __init__(self, r_unit):
        self.r = r_unit; self._h = []

    def _proj(self, module, inp, out):
        def ab(h):
            r = self.r.to(h.device, h.dtype)
            return h - (h @ r).unsqueeze(-1) * r
        return (ab(out[0]),) + out[1:] if isinstance(out, tuple) else ab(out)

    def register(self, model):
        self._h.append(model.model.embed_tokens.register_forward_hook(self._proj))
        for L in model.model.layers:
            self._h.append(L.self_attn.register_forward_hook(self._proj))
            self._h.append(L.mlp.register_forward_hook(self._proj))
        return self

    def remove(self):
        for h in self._h: h.remove()
        self._h = []


class AdditionHook:
    def __init__(self, r_raw, layer_idx):
        self.r = r_raw; self.layer_idx = layer_idx; self._h = None

    def _add(self, module, inp, out):
        if isinstance(out, tuple):
            return (out[0] + self.r.to(out[0].device, out[0].dtype),) + out[1:]
        return out + self.r.to(out.device, out.dtype)

    def register(self, model):
        self._h = model.model.layers[self.layer_idx].register_forward_hook(self._add)
        return self

    def remove(self):
        if self._h: self._h.remove(); self._h = None


@torch.inference_mode()
def gen(model, tok, sys, user):
    msgs = ([{"role": "system", "content": sys}] if sys else []) + [{"role": "user", "content": user}]
    ids = tok.apply_chat_template(msgs, return_tensors="pt", add_generation_prompt=True).to(model.device)
    out = model.generate(ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()


@torch.inference_mode()
def postresp_rep(model, tok, sys, user, layer):
    """Mean-pooled response-token L16 rep: generate, then re-run full seq capturing HS."""
    msgs = ([{"role": "system", "content": sys}] if sys else []) + [{"role": "user", "content": user}]
    pre = tok.apply_chat_template(msgs, return_tensors="pt", add_generation_prompt=True).to(model.device)
    out = model.generate(pre, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, pad_token_id=tok.eos_token_id)
    full = out  # [1, T]
    hs = model(full, output_hidden_states=True, use_cache=False).hidden_states[layer + 1][0]  # [T,D]
    resp = hs[pre.shape[1]:].float().mean(0).cpu()
    return resp


def fit_direction(model, tok, prompts, sys_with, sys_without, layer):
    W, WO = [], []
    for p in prompts:
        W.append(postresp_rep(model, tok, sys_with, p, layer))
        WO.append(postresp_rep(model, tok, sys_without, p, layer))
    w = torch.stack(W).mean(0) - torch.stack(WO).mean(0)
    return {"raw": w, "unit": normalize(w), "raw_norm": float(w.norm())}


def n_tokens(text, tok):
    return len(tok(text, add_special_tokens=False)["input_ids"])


def is_french(text):
    try:
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 0
        return detect(text) == "fr" if text.strip() else False
    except Exception:
        # heuristic fallback
        fr = sum(w in text.lower() for w in [" le ", " la ", " les ", " un ", " une ", " est ", " et ", " je ", " vous ", " pour ", " dans ", "ç", "é", "è", "à"])
        return fr >= 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="data/alpaca_eval.json")
    ap.add_argument("--out-dir", default="experiments/output_content_control_v1")
    ap.add_argument("--n-fit", type=int, default=40)
    ap.add_argument("--n-eval", type=int, default=50)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    allp = [x["instruction"] for x in json.load(open(args.prompts))]
    fit_p = allp[:args.n_fit]
    eval_p = allp[args.n_fit:args.n_fit + args.n_eval]

    print("Loading model...", flush=True)
    model, tok = load_model(device)
    g = torch.Generator().manual_seed(SEED)

    out_path = f"{args.out_dir}/generations.jsonl"
    done = set()
    if os.path.exists(out_path):
        for l in open(out_path):
            try:
                r = json.loads(l); done.add((r["prop"], r["prompt_id"], r["arm"]))
            except Exception:
                pass
    fout = open(out_path, "a")

    summary = {}
    for prop, cfg in PROPS.items():
        print(f"\n=== {prop}: fitting direction ===", flush=True)
        d = fit_direction(model, tok, fit_p, cfg["sys_with"], cfg["sys_without"], LAYER)
        r_unit, r_raw = d["unit"], d["raw"]
        r_rand = normalize(torch.randn(r_unit.shape[0], generator=g).float())
        r_rand_raw = r_rand * d["raw_norm"]
        print(f"  raw_norm={d['raw_norm']:.2f}", flush=True)

        def measure(resp):
            return {"is_french": is_french(resp), "n_tokens": n_tokens(resp, tok)}

        def run(arm, sys, hook_factory):
            for i, p in enumerate(eval_p):
                if (prop, i, arm) in done:
                    continue
                hook = hook_factory() if hook_factory else None
                if hook: hook.register(model)
                try:
                    resp = gen(model, tok, sys, p)
                finally:
                    if hook: hook.remove()
                fout.write(json.dumps({"prop": prop, "prompt_id": i, "arm": arm,
                                       "response": resp, **measure(resp)}) + "\n")
                fout.flush()
            print(f"  [{prop}/{arm}] done", flush=True)

        # ADD arm: on 'without' prompts, add direction -> property appears?
        run("add_baseline", cfg["sys_without"], None)
        run("add_prop", cfg["sys_without"], lambda: AdditionHook(r_raw, LAYER))
        run("add_random", cfg["sys_without"], lambda: AdditionHook(r_rand_raw, LAYER))
        # ABLATE arm: on 'with' prompts, ablate direction -> property disappears?
        run("abl_baseline", cfg["sys_with"], None)
        run("abl_prop", cfg["sys_with"], lambda: AblationHooks(r_unit))
        run("abl_random", cfg["sys_with"], lambda: AblationHooks(r_rand))
        summary[prop] = {"raw_norm": d["raw_norm"]}

    fout.close()
    json.dump(summary, open(f"{args.out_dir}/meta.json", "w"), indent=2)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
