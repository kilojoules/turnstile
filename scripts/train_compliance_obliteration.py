#!/usr/bin/env python
"""LoRA+SFT trainer with compliance-probe obliteration penalty.

One-time experiment. Trains a LoRA adapter on Llama-3.1-8B-Instruct over the
existing victim responses from the 11-run probe corpus (~9,400 convs,
~30k per-turn assistant responses). The loss has two terms:

    L = L_SFT + beta * (soft_AUC(probe(h_L16), y) - 0.5) ** 2

L_SFT is next-token CE on the assistant tokens of the current turn (same
masking convention as turnstile.model_utils.train_lora_multiturn). The
probe is a closed-form ridge LR refit every batch on the L16 residuals
taken at the pre-generation position (same position the canonical
compliance probe was fit on). The soft-AUC is a Wilcoxon-style pairwise
sigmoid so the gradient flows cleanly back through H -> LoRA params.

Alongside this, a long-lived 2-hidden-layer MLP audit probe is trained
on detached residuals as a stronger (nonlinear) leakage monitor. It
does not affect the model's gradient -- its AUC over training is logged
as a diagnostic.

Class-balanced batches (default B+ = B- = 8) so the per-batch ridge fit
is meaningful.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from turnstile.model_utils import _compute_assistant_spans


# ---------------------------------------------------------------------------
# Canonical sources / paths
# ---------------------------------------------------------------------------

POOL_SOURCES = [
    "selfplay_jbb_v1",
    "stealth_s42", "stealth_hard_s456",
    "control_s42", "control_hard_s456",
    "authority_dpo", "reward_dpo",
    "incrementalism_dpo", "urgency_dpo",
    "frozen_v1", "urgency_v1", "stealth_jbb_v1",
]

VICTIM_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
PROBE_LAYER = 16
HIDDEN_DIM = 4096
# Despite the filename, this is the post-migration Llama-judged per-turn
# JBB compliance file (overwritten by the Llama re-judge). Schema keys
# `qwen_unsafe` / `per_turn_unsafe` are historical; values are Llama.
LABELS_REL = "experiments/steering_v3/layer_sweep/qwen_per_turn_compliance.jsonl"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_examples(root, sources, labels_path, compliance_labels_path,
                  label_mode="compliance"):
    """Flatten the corpus into per-turn examples joined to a binary label.

    label_mode='compliance': per-turn JBB unsafe from
        qwen_per_turn_compliance.jsonl (one label per turn).
    label_mode='harm': conv-level Stage-B Likert>=4 from
        stage_b_scores_llama.jsonl (broadcast to all turns of the conv).

    Every example also gets a `sft_label` field (per-turn corpus compliance)
    used by --corpus-compliant-only to gate the SFT loss independently of
    the probe label.
    """
    # Always load per-turn compliance labels for the sft_label field.
    compliance_labels = {}
    with open(compliance_labels_path) as f:
        for line in f:
            rec = json.loads(line)
            key = (rec["source"], rec["round"], rec["idx"])
            compliance_labels[key] = rec.get("per_turn_unsafe") or []

    probe_labels = {}
    if label_mode == "compliance":
        probe_labels = compliance_labels
    elif label_mode == "harm":
        with open(labels_path) as f:
            for line in f:
                rec = json.loads(line)
                key = (rec["source"], rec["round"], rec["idx"])
                rating = rec.get("rating")
                if rating is None:
                    continue
                probe_labels[key] = [int(rating >= 4)] * 5
    else:
        raise ValueError(f"unknown label_mode {label_mode!r}")

    examples = []
    for src in sources:
        rounds_dir = Path(root) / "experiments" / src / "rounds"
        if not rounds_dir.is_dir():
            print(f"  [skip] {src}: no rounds dir at {rounds_dir}")
            continue
        for rf in sorted(rounds_dir.glob("round_*.jsonl")):
            rnum = int(rf.stem.replace("round_", ""))
            with open(rf) as f:
                for idx, line in enumerate(f):
                    rec = json.loads(line)
                    per_turn_probe = probe_labels.get((src, rnum, idx))
                    if per_turn_probe is None:
                        continue
                    per_turn_sft = compliance_labels.get((src, rnum, idx)) or []
                    msgs = rec.get("conversation", [])
                    asst_indices = [i for i, m in enumerate(msgs)
                                    if m["role"] == "assistant"]
                    for t, ai in enumerate(asst_indices):
                        if t >= len(per_turn_probe):
                            break
                        sft_lbl = int(bool(per_turn_sft[t])) if t < len(per_turn_sft) else 0
                        examples.append({
                            "source": src,
                            "round": rnum,
                            "idx": idx,
                            "turn": t,
                            "messages": msgs[: ai + 1],
                            "label": int(bool(per_turn_probe[t])),
                            "sft_label": sft_lbl,
                        })
    return examples


def stratified_split(examples, eval_frac=0.1, seed=0):
    """Group-aware split: hold out whole (source, round) groups for eval."""
    rng = random.Random(seed)
    by_group = {}
    for i, ex in enumerate(examples):
        by_group.setdefault((ex["source"], ex["round"]), []).append(i)
    groups = list(by_group.keys())
    rng.shuffle(groups)
    n_eval = max(1, int(len(groups) * eval_frac))
    eval_groups = set(groups[:n_eval])
    train_idx, eval_idx = [], []
    for g, ids in by_group.items():
        (eval_idx if g in eval_groups else train_idx).extend(ids)
    return [examples[i] for i in train_idx], [examples[i] for i in eval_idx]


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def tokenize_example(tokenizer, ex, max_length=1024):
    """Return (tokens, (asst_start, asst_end), probe_pos) or None.

    probe_pos = asst_start - 1 is the last token of the user prompt +
    generation-prompt suffix, which is the position the canonical L16
    compliance probe reads.
    """
    tokens, spans = _compute_assistant_spans(tokenizer, ex["messages"])
    if not spans:
        return None
    start, end = spans[-1]
    if tokens.shape[0] > max_length:
        drop = tokens.shape[0] - max_length
        tokens = tokens[drop:]
        start -= drop
        end -= drop
        if start < 1:
            return None
    probe_pos = start - 1
    if probe_pos < 0 or end > tokens.shape[0] or end <= start:
        return None
    return tokens, (start, end), probe_pos


def collate_batch(tokenizer, examples, max_length=1024):
    rows = []
    for ex in examples:
        out = tokenize_example(tokenizer, ex, max_length=max_length)
        if out is not None:
            rows.append((out, ex["label"], ex.get("sft_label", ex["label"])))
    if not rows:
        return None
    max_len = max(tokens.shape[0] for (tokens, _, _), _, _ in rows)
    B = len(rows)
    pad_id = tokenizer.pad_token_id
    input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    sft_labels = torch.full((B, max_len), -100, dtype=torch.long)
    probe_pos = torch.zeros(B, dtype=torch.long)
    y = torch.zeros(B, dtype=torch.long)
    sft_y = torch.zeros(B, dtype=torch.long)
    for j, ((tokens, (s, e), pp), lab, sft_lab) in enumerate(rows):
        L = tokens.shape[0]
        input_ids[j, :L] = tokens
        sft_labels[j, s:e] = tokens[s:e]
        probe_pos[j] = pp
        y[j] = lab
        sft_y[j] = sft_lab
    attention = (input_ids != pad_id).long()
    return input_ids, attention, sft_labels, probe_pos, y, sft_y


# ---------------------------------------------------------------------------
# Closed-form ridge probe + soft-AUC
# ---------------------------------------------------------------------------

def closed_form_ridge_logits(H, y, lam=1.0):
    """Ridge LR via dual (Gram) form. Differentiable through H.

    For B << D we solve in B-space:  (K + lam*I) alpha = y_c  with
    K = H_c H_c^T, then logits z = K alpha.  O(B^3) instead of O(D^3).
    """
    H_c = H - H.mean(dim=0, keepdim=True)
    y_f = y.float()
    y_c = y_f - y_f.mean()
    K = H_c @ H_c.T
    B = K.shape[0]
    A = K + lam * torch.eye(B, device=H.device, dtype=H.dtype)
    alpha = torch.linalg.solve(A, y_c.unsqueeze(-1)).squeeze(-1)
    return K @ alpha


def soft_auc(z, y, tau=1.0):
    """Wilcoxon soft-AUC: mean of sigmoid((z+ - z-)/tau) over pos/neg pairs."""
    pos = z[y == 1]
    neg = z[y == 0]
    if pos.numel() == 0 or neg.numel() == 0:
        return torch.tensor(0.5, device=z.device, dtype=z.dtype)
    diffs = pos.unsqueeze(1) - neg.unsqueeze(0)
    return torch.sigmoid(diffs / tau).mean()


# ---------------------------------------------------------------------------
# Adversarial MLP audit probe (monitoring only; no gradient to base model)
# ---------------------------------------------------------------------------

class AuditProbe(nn.Module):
    def __init__(self, dim=HIDDEN_DIM, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_class_balanced(pos_pool, neg_pool, n_pos, n_neg, rng):
    b = [pos_pool[rng.randrange(len(pos_pool))] for _ in range(n_pos)]
    b += [neg_pool[rng.randrange(len(neg_pool))] for _ in range(n_neg)]
    rng.shuffle(b)
    return b


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, tokenizer, eval_ex, args, audit, max_batches=8):
    model.eval()
    pos = [e for e in eval_ex if e["label"] == 1]
    neg = [e for e in eval_ex if e["label"] == 0]
    if not pos or not neg:
        model.train()
        return {"sft_loss": None, "ridge_auc": None, "audit_auc": None,
                "note": "single-class eval"}
    rng = random.Random(0)
    sft_losses, ridge_aucs, audit_aucs = [], [], []
    for _ in range(max_batches):
        batch = sample_class_balanced(pos, neg, args.batch_pos, args.batch_neg, rng)
        coll = collate_batch(tokenizer, batch, max_length=args.max_length)
        if coll is None:
            continue
        input_ids, attention, sft_labels, probe_pos, y, sft_y = (
            t.to(model.device) for t in coll
        )
        if getattr(args, "corpus_compliant_only", False) and args.label_mode == "compliance":
            sft_labels = sft_labels.clone()
            sft_labels[sft_y == 0] = -100
        out = model(input_ids=input_ids, attention_mask=attention,
                    labels=sft_labels, output_hidden_states=True)
        sft_losses.append(float(out.loss))
        h = out.hidden_states[args.probe_layer]
        B = h.shape[0]
        idx = probe_pos.view(B, 1, 1).expand(B, 1, h.shape[-1])
        h_probe = h.gather(1, idx).squeeze(1).to(torch.float32)
        z = closed_form_ridge_logits(h_probe, y, lam=args.ridge_lambda)
        ridge_aucs.append(float(soft_auc(z, y, tau=args.auc_tau)))
        a_logits = audit(h_probe.float())
        audit_aucs.append(float(soft_auc(a_logits, y, tau=args.auc_tau)))
    model.train()
    return {
        "sft_loss": float(np.mean(sft_losses)) if sft_losses else None,
        "ridge_auc": float(np.mean(ridge_aucs)) if ridge_aucs else None,
        "audit_auc": float(np.mean(audit_aucs)) if audit_aucs else None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=".")
    ap.add_argument("--output", type=str, required=True,
                    help="Run dir, e.g. experiments/compliance_obliteration_v1")
    ap.add_argument("--sources", nargs="+", default=POOL_SOURCES)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-pos", type=int, default=8)
    ap.add_argument("--batch-neg", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--beta", type=float, default=4.0,
                    help="Weight on (soft_auc - 0.5)^2")
    ap.add_argument("--ridge-lambda", type=float, default=1.0)
    ap.add_argument("--auc-tau", type=float, default=1.0)
    ap.add_argument("--audit-lr", type=float, default=1e-3)
    ap.add_argument("--audit-inner-steps", type=int, default=3)
    ap.add_argument("--audit-buffer", type=int, default=2048)
    ap.add_argument("--audit-sample", type=int, default=64)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--probe-layer", type=int, default=PROBE_LAYER)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quant-4bit", action="store_true",
                    help="Use 4-bit BnB quantization (bf16 by default)")
    ap.add_argument("--grad-checkpoint", action="store_true", default=True,
                    help="Enable gradient checkpointing (default ON; required "
                         "for batch 16 x seq 1024 bf16 on 80GB)")
    ap.add_argument("--no-grad-checkpoint", dest="grad_checkpoint",
                    action="store_false")
    ap.add_argument("--label-mode", choices=["compliance", "harm"],
                    default="compliance")
    ap.add_argument("--labels-path", default=None,
                    help="Override default labels file for the chosen mode")
    ap.add_argument("--compliance-labels-path", default=None,
                    help="Override compliance labels used for sft_label "
                         "(defaults to LABELS_REL)")
    ap.add_argument("--corpus-compliant-only", action="store_true", default=False,
                    help="Restrict SFT corpus to corpus-compliant turns only. "
                         "Compliance mode: mask neg rows from SFT loss but keep "
                         "them in the probe batch. Harm mode: filter the entire "
                         "training pool to compliance-positive examples.")
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    adapter_path = Path(args.output) / "adapter"
    log_path = Path(args.output) / "train_log.jsonl"
    with open(Path(args.output) / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ----- data -----
    if args.labels_path:
        labels_path = os.path.join(args.root, args.labels_path)
    elif args.label_mode == "compliance":
        labels_path = os.path.join(args.root, LABELS_REL)
    elif args.label_mode == "harm":
        labels_path = os.path.join(args.root, "working/uplift/stage_b_scores_llama.jsonl")
    if args.compliance_labels_path:
        compliance_labels_path = os.path.join(args.root, args.compliance_labels_path)
    else:
        compliance_labels_path = os.path.join(args.root, LABELS_REL)
    print(f"[data] mode={args.label_mode}  sources={len(args.sources)}  "
          f"labels={labels_path}  compliant_only={args.corpus_compliant_only}")
    examples = load_examples(args.root, args.sources, labels_path,
                             compliance_labels_path=compliance_labels_path,
                             label_mode=args.label_mode)
    train_ex, eval_ex = stratified_split(examples, eval_frac=0.1, seed=args.seed)
    if args.corpus_compliant_only and args.label_mode == "harm":
        # Filter entire pool to corpus-compliant examples so both SFT and probe
        # operate within the compliant subset.
        before = len(train_ex)
        train_ex = [e for e in train_ex if e["sft_label"] == 1]
        eval_ex  = [e for e in eval_ex  if e["sft_label"] == 1]
        print(f"[data] corpus_compliant_only: train {before} -> {len(train_ex)}")
    n_pos = sum(1 for e in train_ex if e["label"] == 1)
    n_neg = len(train_ex) - n_pos
    print(f"[data] train={len(train_ex)} ({n_pos}+/{n_neg}-)  eval={len(eval_ex)}")
    pos_pool = [e for e in train_ex if e["label"] == 1]
    neg_pool = [e for e in train_ex if e["label"] == 0]
    if not pos_pool or not neg_pool:
        raise RuntimeError("Train pool lacks one class; check labels file.")

    # ----- tokenizer -----
    tokenizer = AutoTokenizer.from_pretrained(VICTIM_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ----- model + LoRA -----
    print(f"[model] {VICTIM_MODEL}  quant_4bit={args.quant_4bit}")
    if args.quant_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            VICTIM_MODEL, quantization_config=bnb,
            device_map="auto", torch_dtype=torch.bfloat16,
        )
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=False,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            VICTIM_MODEL, device_map="auto", torch_dtype=torch.bfloat16,
        )
    model.enable_input_require_grads()  # keep grad flowing through embeddings

    lora_cfg = LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    # Gradient checkpointing is required to fit batch 16 x seq 1024 at bf16
    # on 80GB without OOMing the attention activations. ~25% slower.
    model.config.use_cache = False
    if args.grad_checkpoint:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    model.train()

    # ----- audit probe (fp32 for stable Adam) -----
    audit = AuditProbe(dim=HIDDEN_DIM).to(model.device, dtype=torch.float32)
    audit_opt = torch.optim.AdamW(audit.parameters(), lr=args.audit_lr)

    # ----- main optimizer -----
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)

    audit_buffer_h = []  # list of (D,) cpu fp32 tensors
    audit_buffer_y = []

    def update_audit():
        if len(audit_buffer_h) < args.audit_sample:
            return None
        N = min(args.audit_sample, len(audit_buffer_h))
        last_loss = None
        for _ in range(args.audit_inner_steps):
            ids = np.random.choice(len(audit_buffer_h), size=N, replace=False)
            h = torch.stack([audit_buffer_h[i] for i in ids]).to(
                model.device, dtype=torch.float32,
            )
            yy = torch.tensor(
                [audit_buffer_y[i] for i in ids],
                device=model.device, dtype=torch.float32,
            )
            logits = audit(h)
            loss = F.binary_cross_entropy_with_logits(logits, yy)
            audit_opt.zero_grad()
            loss.backward()
            audit_opt.step()
            last_loss = float(loss)
        return last_loss

    # ----- training loop -----
    print(f"[train] steps={args.steps}  batch={args.batch_pos}+{args.batch_neg}  "
          f"beta={args.beta}  layer={args.probe_layer}")
    t0 = time.time()
    log_fh = open(log_path, "w")

    for step in range(args.steps):
        batch_ex = sample_class_balanced(
            pos_pool, neg_pool, args.batch_pos, args.batch_neg, rng,
        )
        coll = collate_batch(tokenizer, batch_ex, max_length=args.max_length)
        if coll is None:
            continue
        input_ids, attention, sft_labels, probe_pos, y, sft_y = (
            t.to(model.device) for t in coll
        )
        if args.corpus_compliant_only and args.label_mode == "compliance":
            sft_labels = sft_labels.clone()
            sft_labels[sft_y == 0] = -100

        out = model(
            input_ids=input_ids, attention_mask=attention,
            labels=sft_labels, output_hidden_states=True,
        )
        L_sft = out.loss
        h_layer = out.hidden_states[args.probe_layer]  # (B, L, D)
        B = h_layer.shape[0]
        idx = probe_pos.view(B, 1, 1).expand(B, 1, h_layer.shape[-1])
        h_probe = h_layer.gather(1, idx).squeeze(1).to(torch.float32)

        z = closed_form_ridge_logits(h_probe, y, lam=args.ridge_lambda)
        auc = soft_auc(z, y, tau=args.auc_tau)
        L_obl = args.beta * (auc - 0.5) ** 2
        loss = L_sft + L_obl

        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        optimizer.zero_grad()

        # audit probe inner update on detached residuals
        with torch.no_grad():
            for j in range(B):
                audit_buffer_h.append(h_probe[j].detach().to("cpu"))
                audit_buffer_y.append(int(y[j].item()))
            while len(audit_buffer_h) > args.audit_buffer:
                audit_buffer_h.pop(0)
                audit_buffer_y.pop(0)
        audit_loss = update_audit()
        with torch.no_grad():
            a_logits = audit(h_probe.detach().float())
            audit_auc_val = float(soft_auc(a_logits, y, tau=args.auc_tau))

        rec = {
            "step": step,
            "sft_loss": float(L_sft),
            "obl_loss": float(L_obl),
            "ridge_auc": float(auc),
            "audit_loss": audit_loss,
            "audit_auc": audit_auc_val,
            "elapsed": time.time() - t0,
        }
        log_fh.write(json.dumps(rec) + "\n")
        log_fh.flush()

        if (step + 1) % 10 == 0:
            print(f"  step {step+1}/{args.steps}  "
                  f"sft={float(L_sft):.3f}  "
                  f"obl={float(L_obl):.4f}  "
                  f"ridge_auc={float(auc):.3f}  "
                  f"audit_auc={audit_auc_val:.3f}")

        if (step + 1) % args.eval_every == 0 and eval_ex:
            evm = evaluate(model, tokenizer, eval_ex, args, audit)
            evm["step"] = step + 1
            log_fh.write(json.dumps({"_eval": True, **evm}) + "\n")
            log_fh.flush()
            print(f"  [eval@{step+1}] {evm}")

        if (step + 1) % args.save_every == 0:
            model.save_pretrained(adapter_path)
            torch.save(audit.state_dict(), Path(args.output) / "audit_probe.pt")

    model.save_pretrained(adapter_path)
    torch.save(audit.state_dict(), Path(args.output) / "audit_probe.pt")
    log_fh.close()
    print(f"[done] saved -> {args.output}")


if __name__ == "__main__":
    main()
