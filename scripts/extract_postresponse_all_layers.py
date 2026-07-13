"""Extract per-layer residuals at BOTH loci (pre-response 'input' and response-mean
'output'/posterior) for the harm and compliance corpora, to rebuild
auc_by_layer_postaudit at the POSTERIOR locus with a matched-corpus prior baseline.

Recipe mirrors build_directions.turn_reps / extract_harm_both_loci exactly:
  pre  = apply_chat_template(history, add_generation_prompt=True)
  full = pre + response
  input  = hidden_states[HS(L)][p-1]            (last prompt token, pre-response)
  output = hidden_states[HS(L)][p:].mean(0)      (mean over the response tokens)
  HS(L) = L+1   (hidden_states[0]=embeddings)

RESUMABLE (this box silently kills the process intermittently):
  - HARM first (289 passes) -> reps_harm.npz; skipped entirely if it already exists.
  - COMPLIANCE resumes from reps_comp.npz: seeds from the last checkpoint and SKIPS
    already-processed conversation indices, so an outer restart loop just continues.
  - checkpoint every 20 records; per-pass try/except + MAXTOK length guard.
  - prints 'ALL DONE' only when compliance is fully processed.
"""
import json, os, hashlib, time
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ROOT = os.environ.get("PR_ROOT", ".")
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
LAYERS = [0, 4, 8, 12, 16, 20, 24, 28, 31]
HS = {L: L + 1 for L in LAYERS}
MAXTOK = int(os.environ.get("PR_MAXTOK", "3200"))
OUT = os.path.join(ROOT, "experiments/postresponse_alllayer")
os.makedirs(OUT, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
HARM_P = os.path.join(OUT, "reps_harm.npz")
COMP_P = os.path.join(OUT, "reps_comp.npz")


def ghash(s):
    return int(hashlib.md5(str(s).encode()).hexdigest()[:8], 16)


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map=dev).eval()

    @torch.inference_mode()
    def turn_reps(history, response):
        pre = tok.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
        p = tok(pre, return_tensors="pt").input_ids.shape[1]
        ids = tok(pre + response, return_tensors="pt").input_ids
        if ids.shape[1] > MAXTOK:
            return None
        hs = model(ids.to(dev), output_hidden_states=True, use_cache=False).hidden_states
        ins, outs = {}, {}
        for L in LAYERS:
            h = hs[HS[L]][0].float().cpu().numpy()
            ins[L] = h[p - 1]
            outs[L] = h[p:].mean(0) if h.shape[0] > p else h[-1]
        del hs
        return ins, outs

    def safe(history, response):
        try:
            return turn_reps(history, response)
        except RuntimeError as e:
            print(f"    [skip pass] {str(e)[:80]}", flush=True)
            if dev == "cuda":
                torch.cuda.empty_cache()
            return None

    # ---------- HARM ----------
    # DEFAULT = FULL-600 corpus (all valid-rating turns, incl. refused) so the per-fold MD
    # matches the steered harm_dm_llama recipe [norm(mu(rating>=4)-mu(rating<=2)), L16-out].
    # PR_HARM_WINS_ONLY=1 restores the wins-only variant. Always re-runs (checkpoints for pulls).
    WINS_ONLY = os.environ.get("PR_HARM_WINS_ONLY", "0") == "1"
    t0 = time.time()
    cand = {(c["source"], c["round"], c["idx"]): c for c in map(json.loads, open(os.path.join(ROOT, "working/uplift/stage_b_candidates.jsonl")))}
    Hin = {L: [] for L in LAYERS}; Hout = {L: [] for L in LAYERS}; hrl, hrq, hgrp = [], [], []

    def hckpt():
        s = {"layers": np.array(LAYERS), "harm_rl": np.array(hrl), "harm_rq": np.array(hrq), "harm_grp": np.array(hgrp)}
        for L in LAYERS:
            s[f"harm_in_L{L}"] = np.stack(Hin[L]); s[f"harm_out_L{L}"] = np.stack(Hout[L])
        np.savez(HARM_P, **s)

    for s in map(json.loads, open(os.path.join(ROOT, "working/uplift/stage_b_scores_llama.jsonl"))):
        if WINS_ONLY and not s.get("unsafe"):
            continue
        lrr, qr = s.get("rating"), s.get("qwen_rating")
        if not isinstance(lrr, int) or not isinstance(qr, int):
            continue
        c = cand.get((s["source"], s["round"], s["idx"]))
        if not c:
            continue
        conv = c["conversation"]; ai = [i for i, t in enumerate(conv) if t["role"] == "assistant"]
        tob = s.get("turn_of_breach"); bi = ai[tob] if (isinstance(tob, int) and tob < len(ai)) else (ai[-1] if ai else -1)
        if bi <= 0 or conv[bi - 1]["role"] != "user":
            continue
        r = safe(conv[:bi], conv[bi]["content"])
        if r is None:
            continue
        ins, outs = r
        for L in LAYERS:
            Hin[L].append(ins[L]); Hout[L].append(outs[L])
        hrl.append(lrr); hrq.append(qr); hgrp.append(ghash(c.get("goal", "")))
        if len(hrl) % 100 == 0:
            hckpt(); print(f"  [harm] rows={len(hrl)} {time.time()-t0:.0f}s (ckpt)", flush=True)
    hckpt()
    print(f"[harm] DONE rows={len(hrl)} corpus={'wins' if WINS_ONLY else 'FULL600'} "
          f"llama>=4={sum(v>=4 for v in hrl)} <=2={sum(v<=2 for v in hrl)} -> reps_harm.npz", flush=True)

    if os.environ.get("PR_SKIP_COMP") == "1":
        print("[comp] PR_SKIP_COMP=1 — compliance unchanged, skipping", flush=True)
        print("ALL DONE", flush=True)
        return
    # ---------- COMPLIANCE (resumable) ----------
    qmap = {}
    for l in open(os.path.join(ROOT, "experiments/steering_v3/layer_sweep/qwen_per_turn_compliance.jsonl")):
        r = json.loads(l); qmap[(r["source"], r["round"], r["idx"])] = r.get("per_turn_unsafe")
    rp = os.path.join(ROOT, "experiments/clamp_obliterated_v2/replay_v2_slim.pt")
    if not os.path.exists(rp):
        rp = os.path.join(ROOT, "experiments/clamp_obliterated_v2/replay_v2_full.pt")
    recs = torch.load(rp, weights_only=False)

    Cin = {L: [] for L in LAYERS}; Cout = {L: [] for L in LAYERS}; cyl, cyq, cgrp = [], [], []
    done = set()
    if os.path.exists(COMP_P):
        Z = np.load(COMP_P)
        for L in LAYERS:
            Cin[L] = list(Z[f"comp_in_L{L}"]); Cout[L] = list(Z[f"comp_out_L{L}"])
        cyl = list(Z["comp_yl"]); cyq = list(Z["comp_yq"]); cgrp = list(Z["comp_grp"])
        done = set(int(g) for g in cgrp)
        print(f"[comp] resume: {len(cyl)} rows, {len(done)} records already done", flush=True)

    def ckpt():
        s = {"layers": np.array(LAYERS), "comp_yl": np.array(cyl), "comp_yq": np.array(cyq), "comp_grp": np.array(cgrp)}
        for L in LAYERS:
            s[f"comp_in_L{L}"] = np.stack(Cin[L]); s[f"comp_out_L{L}"] = np.stack(Cout[L])
        np.savez(COMP_P, **s)  # COMP_P already ends in .npz, numpy writes it verbatim

    t1 = time.time()
    for gi, r in enumerate(recs):
        if gi in done:
            continue
        conv = r.get("original_conversation"); labs = r.get("original_per_turn_labels")
        if not conv or not labs:
            done.add(gi); continue
        base = str(r.get("source", "")).split("/")[0]
        ptu = qmap.get((base, r.get("round"), r.get("index")))
        ai = [i for i, t in enumerate(conv) if t["role"] == "assistant"]
        lab_by = {l["turn"]: l for l in labs if l.get("unsafe") is not None}
        for t, aidx in enumerate(ai):
            if t not in lab_by or aidx == 0 or conv[aidx - 1]["role"] != "user":
                continue
            res = safe(conv[:aidx], conv[aidx]["content"])
            if res is None:
                continue
            ins, outs = res
            for L in LAYERS:
                Cin[L].append(ins[L]); Cout[L].append(outs[L])
            cyl.append(int(bool(lab_by[t]["unsafe"])))
            cyq.append(int(bool(ptu[t])) if (ptu is not None and t < len(ptu)) else -1)
            cgrp.append(gi)
        done.add(gi)
        if gi % 20 == 0 and cyl:
            ckpt(); print(f"  [comp] record {gi}/{len(recs)} rows={len(cyl)} {time.time()-t1:.0f}s (ckpt)", flush=True)
    ckpt()
    print(f"[comp] DONE rows={len(cyl)} complied={sum(v==1 for v in cyl)} qwen_labeled={sum(v>=0 for v in cyq)} -> reps_comp.npz", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
