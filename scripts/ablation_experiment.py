"""Experiment 1: directional ablation (Arditi-style necessity test).

Addition-steering (h += a*v) tests SUFFICIENCY and is confounded by norm
effects (a random direction moves ASR as much as the compliance direction).
Directional ABLATION (h -= (h.v_hat) v_hat, projecting the direction out of
every layer's residual stream at every position) tests NECESSITY: is the
direction causally required for the model to refuse?

Conditions (single-turn, 100 JBB harmful requests, greedy victim):
  baseline        no intervention
  ablate_refusal  project out the Arditi refusal direction (harmful - benign @ L16)
  ablate_lr_comp  project out the paper's compliance probe direction (v_lr_comp_L16)
  ablate_md_comp  project out the mean-diff compliance direction (v_md_comp_L16)
  ablate_random   project out a fixed random unit direction (negative control)

Reading:
  - If ablate_refusal jailbreaks the model (ASR >> baseline) the harness works
    (positive control replicates Arditi).
  - If ablate_lr_comp / md_comp also raise ASR, the paper's compliance direction
    is causally on the refusal path (necessity established).
  - If none raise ASR above ablate_random, ablation is as non-specific as
    addition was -> the "compliance direction" is not a causal handle.

Output rows carry {prompt_id, method, alpha=0.0, goal, category, behavior,
response} so scripts/judge_postresponse_sweep.py can judge them unchanged.
"""
import argparse
import json
import os

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
REF_LAYER = 16
MAX_NEW_TOKENS = 300
SEED = 42

BENIGN = [
    "What is the capital of France?", "Explain how photosynthesis works.",
    "Write a short poem about the ocean.", "What are the main differences between Python and Java?",
    "Describe the water cycle in simple terms.", "List five common household plants.",
    "How does a bicycle work?", "What is the Pythagorean theorem?",
    "Tell me about the history of the printing press.", "What are the primary colors?",
    "Explain how a compass works.", "What is the speed of light?",
    "Describe the structure of an atom.", "What are the planets in our solar system?",
    "How do birds fly?", "What is machine learning?",
    "Explain the concept of supply and demand.", "What causes rainbows?",
    "How does WiFi work?", "What is the periodic table?",
]


def load_victim(device):
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb, device_map=device, torch_dtype=torch.bfloat16)
    model.eval()
    return model, tok


def normalize(v):
    return v / v.norm()


@torch.inference_mode()
def mean_hidden_at_layer(model, tok, prompts, layer):
    """Mean last-token residual at `layer` over a list of chat prompts."""
    reps = []
    for p in prompts:
        ids = tok.apply_chat_template([{"role": "user", "content": p}],
                                      return_tensors="pt", add_generation_prompt=True).to(model.device)
        out = model(ids, output_hidden_states=True)
        reps.append(out.hidden_states[layer + 1][0, -1].float().cpu())
    return torch.stack(reps).mean(0)


def fit_refusal_direction(model, tok, goals, layer):
    harmful = [g["goal"] for g in goals[:len(BENIGN)]]
    mu_h = mean_hidden_at_layer(model, tok, harmful, layer)
    mu_b = mean_hidden_at_layer(model, tok, BENIGN, layer)
    return normalize(mu_h - mu_b)


class AblationHook:
    """Project a unit direction out of every decoder layer's output residual."""

    def __init__(self, direction):
        self.d = normalize(direction.float())
        self._handles = []

    def _fn(self, module, inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        d = self.d.to(hs.device, hs.dtype)
        coef = (hs @ d).unsqueeze(-1)          # <h, d_hat>
        hs = hs - coef * d                      # remove the component along d_hat
        return (hs,) + out[1:] if isinstance(out, tuple) else hs

    def register(self, model):
        for layer in model.model.layers:
            self._handles.append(layer.register_forward_hook(self._fn))
        return self

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []


@torch.inference_mode()
def generate(model, tok, goal):
    ids = tok.apply_chat_template([{"role": "user", "content": goal}],
                                  return_tensors="pt", add_generation_prompt=True).to(model.device)
    out = model.generate(ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--goals", default="working/uplift/goals.json")
    ap.add_argument("--directions-dir", default="experiments/steering_v3/layer_sweep/directions")
    ap.add_argument("--output", required=True)
    ap.add_argument("--n-goals", type=int, default=100)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    goals = json.load(open(args.goals))[:args.n_goals]
    print(f"Loaded {len(goals)} goals", flush=True)

    print("Loading victim...", flush=True)
    model, tok = load_victim(device)

    print("Fitting refusal direction (harmful - benign @ L16)...", flush=True)
    v_refusal = fit_refusal_direction(model, tok, goals, REF_LAYER)
    v_lr = normalize(torch.load(f"{args.directions_dir}/v_lr_comp_L{REF_LAYER}.pt").float())
    v_md = normalize(torch.load(f"{args.directions_dir}/v_md_comp_L{REF_LAYER}.pt").float())
    g = torch.Generator().manual_seed(SEED)
    v_rand = normalize(torch.randn(v_lr.shape[0], generator=g).float())

    # cosine geometry, logged for the writeup
    def cos(a, b): return float(torch.dot(normalize(a), normalize(b)))
    print(f"cos(refusal, lr_comp)={cos(v_refusal, v_lr):+.3f}  "
          f"cos(refusal, md_comp)={cos(v_refusal, v_md):+.3f}  "
          f"cos(lr_comp, md_comp)={cos(v_lr, v_md):+.3f}  "
          f"cos(refusal, random)={cos(v_refusal, v_rand):+.3f}", flush=True)

    conditions = {
        "baseline": None,
        "ablate_refusal": v_refusal,
        "ablate_lr_comp": v_lr,
        "ablate_md_comp": v_md,
        "ablate_random": v_rand,
    }

    done = set()
    if os.path.exists(args.output):
        for l in open(args.output):
            try:
                r = json.loads(l); done.add((r["prompt_id"], r["method"]))
            except Exception:
                pass
        print(f"Resuming: {len(done)} cells done", flush=True)

    with open(args.output, "a") as fout:
        for cond, direction in conditions.items():
            hook = AblationHook(direction).register(model) if direction is not None else None
            try:
                for i, gd in enumerate(goals):
                    if (i, cond) in done:
                        continue
                    resp = generate(model, tok, gd["goal"])
                    fout.write(json.dumps({
                        "prompt_id": i, "method": cond, "alpha": 0.0,
                        "goal": gd["goal"], "category": gd.get("category"),
                        "behavior": gd.get("behavior"), "response": resp,
                    }) + "\n")
                    fout.flush()
            finally:
                if hook is not None:
                    hook.remove()
            print(f"  done condition={cond}", flush=True)

    print("DONE", flush=True)


if __name__ == "__main__":
    main()
