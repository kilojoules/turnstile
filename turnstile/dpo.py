"""DPO (Direct Preference Optimization) training for Turnstile adversary.

Instead of SFT on wins only (sparse signal), DPO learns from (chosen, rejected)
pairs: successful jailbreaks vs. failed attacks for the same goal. Every
conversation provides training signal — not just the 3-7% that are wins.

Data format: JSONL with {"chosen": [...], "rejected": [...], "goal": "..."}
where chosen/rejected are adversary-perspective message lists.

Reference: Rafailov et al., "Direct Preference Optimization" (NeurIPS 2023).
"""

import gc
import json
import os
import random
from collections import defaultdict

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training

from turnstile.model_utils import (
    _BNB_CONFIG,
    _inject_chat_template,
    _compute_assistant_spans,
    unload_model,
)


# ---------------------------------------------------------------------------
# DPO pair construction from round data
# ---------------------------------------------------------------------------

def _victim_to_adversary(conversation, goal):
    """Convert victim-perspective conversation to adversary training format."""
    adv_messages = [
        {"role": "system", "content": f"Goal: {goal}"},
        {"role": "user", "content": "Begin the conversation."},
    ]
    for msg in conversation:
        if msg["role"] == "user":
            adv_messages.append({"role": "assistant", "content": msg["content"]})
        elif msg["role"] == "assistant":
            adv_messages.append({
                "role": "user",
                "content": f"Target responded: {msg['content']}\n\nGenerate next message.",
            })
    return adv_messages


def build_dpo_pairs(round_files, per_turn=True):
    """Build DPO pairs from round JSONL files.

    Groups conversations by goal, then pairs each win with each loss.
    If per_turn=True, splits multi-turn conversations into per-turn
    examples before pairing (more pairs, each with a single loss target).

    Returns list of {"chosen": messages, "rejected": messages, "goal": str}.
    """
    convs_by_goal = defaultdict(lambda: {"wins": [], "losses": []})

    for f in round_files:
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                goal = r.get("goal", "")
                if not goal or not r.get("conversation"):
                    continue

                adv_msgs = _victim_to_adversary(r["conversation"], goal)
                bucket = "wins" if r.get("unsafe", False) else "losses"

                if per_turn:
                    # Split into per-turn examples
                    asst_indices = [i for i, m in enumerate(adv_msgs)
                                    if m["role"] == "assistant"]
                    for ai in asst_indices:
                        prefix = adv_msgs[:ai + 1]
                        convs_by_goal[goal][bucket].append(prefix)
                else:
                    convs_by_goal[goal][bucket].append(adv_msgs)

    # Build pairs: for each goal, pair wins with losses
    pairs = []
    for goal, data in convs_by_goal.items():
        if not data["wins"] or not data["losses"]:
            continue
        for chosen in data["wins"]:
            for rejected in data["losses"]:
                # Only pair examples with the same number of messages
                # (same turn depth) for a fair comparison
                if len(chosen) == len(rejected):
                    pairs.append({
                        "chosen": chosen,
                        "rejected": rejected,
                        "goal": goal,
                    })

    random.shuffle(pairs)
    return pairs


# ---------------------------------------------------------------------------
# Log-probability computation
# ---------------------------------------------------------------------------

def _get_logprobs(model, tokenizer, messages, device):
    """Compute sum of log-probabilities over assistant tokens.

    Returns (total_logprob, n_tokens) for the assistant spans.
    """
    tokens, spans = _compute_assistant_spans(tokenizer, messages)
    if not spans:
        return torch.tensor(0.0, device=device), 0

    tokens = tokens[:2048].unsqueeze(0).to(device)
    attention_mask = torch.ones_like(tokens)

    with torch.no_grad() if not model.training else torch.enable_grad():
        outputs = model(input_ids=tokens, attention_mask=attention_mask)
        logits = outputs.logits

    # Shift: logits[t] predicts token[t+1]
    shift_logits = logits[:, :-1, :]
    shift_labels = tokens[:, 1:]
    log_probs = F.log_softmax(shift_logits, dim=-1)

    total_lp = torch.tensor(0.0, device=device)
    n_tokens = 0
    for start, end in spans:
        # For each assistant span, sum log-probs of predicting those tokens
        # Labels are shifted by 1: logits[start-1] predicts token[start]
        for t in range(max(start, 1), min(end, tokens.shape[1])):
            token_id = shift_labels[0, t - 1]
            total_lp = total_lp + log_probs[0, t - 1, token_id]
            n_tokens += 1

    return total_lp, n_tokens


# ---------------------------------------------------------------------------
# DPO Training
# ---------------------------------------------------------------------------

def train_dpo(
    model_id,
    pairs_file,
    adapter_path,
    num_iters=100,
    lr=5e-6,
    beta=0.1,
    lora_rank=8,
    lora_alpha=16,
    target_modules=None,
    chat_template_model=None,
):
    """DPO fine-tune on preference pairs.

    For each (chosen, rejected) pair:
      L = -log sigmoid(beta * (delta_policy - delta_ref))
    where delta = log_prob(chosen) - log_prob(rejected).

    The reference model is the frozen base (no LoRA). The policy model
    is the LoRA adapter being trained.
    """
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]

    print(f"   [DPO] Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if chat_template_model:
        _inject_chat_template(tokenizer, chat_template_model)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=_BNB_CONFIG,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    device = model.device

    # --- Compute reference log-probs (frozen base, no LoRA) ---
    print(f"   [DPO] Computing reference log-probs...")
    pairs = []
    with open(pairs_file) as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))

    if not pairs:
        print("   [DPO] No pairs — skipping.")
        unload_model(model, tokenizer)
        return

    model.eval()
    ref_deltas = []
    for i, pair in enumerate(pairs):
        lp_c, _ = _get_logprobs(model, tokenizer, pair["chosen"], device)
        lp_r, _ = _get_logprobs(model, tokenizer, pair["rejected"], device)
        ref_deltas.append((lp_c - lp_r).detach())
        if (i + 1) % 50 == 0:
            print(f"   [DPO] Ref log-probs: {i+1}/{len(pairs)}")

    print(f"   [DPO] Reference log-probs computed for {len(pairs)} pairs")

    # --- Apply LoRA for policy ---
    model = prepare_model_for_kbit_training(model)
    if os.path.exists(os.path.join(adapter_path, "adapter_model.safetensors")):
        print(f"   [DPO] Resuming from existing adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
    else:
        print(f"   [DPO] Initializing new LoRA adapter (rank={lora_rank})")
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- DPO training loop ---
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    print(f"   [DPO] Training for {num_iters} steps "
          f"(dataset: {len(pairs)} pairs, beta={beta})")

    losses = []
    accuracies = []
    for step in range(num_iters):
        idx = random.randint(0, len(pairs) - 1)
        pair = pairs[idx]
        ref_delta = ref_deltas[idx]

        # Policy log-probs (with gradient)
        lp_c, _ = _get_logprobs(model, tokenizer, pair["chosen"], device)
        lp_r, _ = _get_logprobs(model, tokenizer, pair["rejected"], device)
        policy_delta = lp_c - lp_r

        # DPO loss: -log sigmoid(beta * (policy_delta - ref_delta))
        logit = beta * (policy_delta - ref_delta)
        loss = -F.logsigmoid(logit)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss.item())
        accuracies.append((logit > 0).float().item())

        if (step + 1) % 10 == 0:
            avg_loss = sum(losses[-10:]) / min(10, len(losses[-10:]))
            avg_acc = sum(accuracies[-10:]) / min(10, len(accuracies[-10:]))
            print(f"   Step {step+1}/{num_iters} | "
                  f"Loss: {avg_loss:.4f} | Acc: {avg_acc:.1%}")

    os.makedirs(adapter_path, exist_ok=True)
    model.save_pretrained(adapter_path)
    print(f"   [DPO] Adapter saved to {adapter_path}")

    unload_model(model, tokenizer, optimizer)


# ---------------------------------------------------------------------------
# CLI: build pairs + train
# ---------------------------------------------------------------------------

def main():
    import argparse
    import glob

    parser = argparse.ArgumentParser(description="DPO training for adversary")
    parser.add_argument("--round-dirs", nargs="+", required=True,
                        help="Directories containing round_*.jsonl files")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for pairs + adapter")
    parser.add_argument("--model-id", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--chat-template-model", default=None)
    parser.add_argument("--num-iters", type=int, default=100)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--per-turn", action="store_true", default=True)
    parser.add_argument("--no-per-turn", action="store_true")
    args = parser.parse_args()

    per_turn = not args.no_per_turn

    # Collect round files
    round_files = []
    for d in args.round_dirs:
        round_files.extend(sorted(glob.glob(os.path.join(d, "round_*.jsonl"))))
    print(f"Found {len(round_files)} round files")

    # Build pairs
    os.makedirs(args.output_dir, exist_ok=True)
    pairs = build_dpo_pairs(round_files, per_turn=per_turn)
    print(f"Built {len(pairs)} DPO pairs (per_turn={per_turn})")

    pairs_file = os.path.join(args.output_dir, "dpo_pairs.jsonl")
    with open(pairs_file, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    print(f"Saved to {pairs_file}")

    # Train
    adapter_path = os.path.join(args.output_dir, "adapters")
    train_dpo(
        model_id=args.model_id,
        pairs_file=pairs_file,
        adapter_path=adapter_path,
        num_iters=args.num_iters,
        lr=args.lr,
        beta=args.beta,
        chat_template_model=args.chat_template_model,
    )


if __name__ == "__main__":
    main()
