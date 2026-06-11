"""H5: Fit an Arditi-style refusal direction on (raw harmful goal) vs
(AlpacaEval benign) inputs.

Mirrors Arditi 2024:
  - Pass each prompt through the victim once
  - Capture HS at the last user-message token (post-chat-template)
  - mu_harm = mean(HS_harm); mu_benign = mean(HS_benign)
  - direction = mu_harm - mu_benign, normalized to unit length

This is the "refusal direction": pointing TOWARD a representation of
"this input is harmful". Ablating it across all layers should bypass
refusal (Arditi figures 1-2). Adding it to a benign prompt should
induce refusal (Arditi figure 3).

Output: directions.pt mapping label -> {direction, info}.
"""

import argparse
import json
import os
import time

import torch

from turnstile.model_utils import load_model, unload_model


def extract_hs_at_layers(model, tokenizer, prompts, layers, device="cuda"):
    by_layer = {L: [] for L in layers}
    for prompt in prompts:
        chat = [{"role": "user", "content": prompt}]
        templated = tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(templated, return_tensors="pt").to(device)
        with torch.inference_mode():
            out = model(**inputs, output_hidden_states=True, use_cache=False)
        for L in layers:
            h = out.hidden_states[L + 1][0, -1, :].detach().to("cpu", torch.float32)
            by_layer[L].append(h)
        del out
    return {L: torch.stack(by_layer[L]) for L in layers}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--harmful-prompts-json", required=True,
                   help="JSON {test:[{harmful_breach,goal,record_idx}]} — uses 'harmful_breach' field")
    p.add_argument("--benign-prompts-json", required=True,
                   help="JSON {benign:[{prompt}]} from AlpacaEval (single_prompt_inputs.json works)")
    p.add_argument("--layers", type=int, nargs="+", default=[16])
    p.add_argument("--n-train", type=int, default=80,
                   help="How many of each side to use for fitting (rest is held out)")
    p.add_argument("--out-pt", required=True)
    p.add_argument("--victim-model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--seed", type=int, default=13)
    args = p.parse_args()

    inp_h = json.load(open(args.harmful_prompts_json))
    test = inp_h["test"]
    harmful_prompts = [x["harmful_breach"] for x in test[:args.n_train]]

    inp_b = json.load(open(args.benign_prompts_json))
    benign_prompts = [x["prompt"] for x in inp_b["benign"][:args.n_train]]

    print(f"fitting on {len(harmful_prompts)} harmful + {len(benign_prompts)} benign prompts")
    print(f"layers: {args.layers}")

    print(f"loading victim {args.victim_model}...")
    vic_model, vic_tok = load_model(args.victim_model)

    t0 = time.time()
    hs_h = extract_hs_at_layers(vic_model, vic_tok, harmful_prompts, args.layers)
    hs_b = extract_hs_at_layers(vic_model, vic_tok, benign_prompts, args.layers)
    print(f"  extraction took {time.time()-t0:.0f}s")

    out = {}
    for L in args.layers:
        mu_h = hs_h[L].mean(dim=0)
        mu_b = hs_b[L].mean(dim=0)
        w = mu_h - mu_b
        raw_norm = float(w.norm())
        d = w / max(raw_norm, 1e-9)
        median_h = float(torch.norm(hs_h[L], dim=1).median())
        sep_h = float((hs_h[L] @ d).mean() - (hs_b[L] @ d).mean())
        out[f"L{L}_arditi_meandiff"] = {
            "direction": d,
            "info": {"method": "arditi_meandiff",
                     "raw_norm": raw_norm,
                     "n_harmful": len(harmful_prompts),
                     "n_benign": len(benign_prompts),
                     "centroid_separation": sep_h},
            "median_h": median_h,
        }
        print(f"L{L} arditi_meandiff  raw_norm={raw_norm:.3f}  centroid_sep={sep_h:+.3f}  median_h={median_h:.2f}")

    os.makedirs(os.path.dirname(args.out_pt), exist_ok=True)
    torch.save(out, args.out_pt)
    print(f"wrote {args.out_pt}")
    unload_model(vic_model, vic_tok)


if __name__ == "__main__":
    main()
