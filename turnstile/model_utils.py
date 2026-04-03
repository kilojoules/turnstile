"""Central HuggingFace / PEFT wrapper for Turnstile pipeline.

Extends REDKWEEN's model_utils with multi-turn conversation support:
- train_lora_multiturn: LoRA training with loss on all assistant tokens
- _compute_assistant_spans: helper for multi-turn label masking

All model loading/inference/training goes through this module.
"""

import gc
import json
import os
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training


# ---------------------------------------------------------------------------
# Quantization config (shared across all loads)
# ---------------------------------------------------------------------------
_BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)


def _inject_chat_template(tokenizer, template_model_id):
    """Copy the chat template from an Instruct model into a base tokenizer.

    Base models (e.g. Llama-3.2-3B) ship without a chat template, so
    apply_chat_template would fail. We borrow the template from the
    corresponding Instruct variant so the base model can be trained on
    and generate chat-formatted text.
    """
    if tokenizer.chat_template is not None:
        return  # already has a template
    instruct_tok = AutoTokenizer.from_pretrained(template_model_id)
    if instruct_tok.chat_template is None:
        raise ValueError(
            f"Template model {template_model_id} has no chat template either"
        )
    tokenizer.chat_template = instruct_tok.chat_template
    # Also copy any special tokens the instruct tokenizer defines that
    # the base tokenizer might lack (e.g. <|eot_id|>).
    for attr in ("eos_token", "bos_token"):
        instruct_val = getattr(instruct_tok, attr, None)
        base_val = getattr(tokenizer, attr, None)
        if instruct_val and not base_val:
            setattr(tokenizer, attr, instruct_val)
    print(f"   [Chat template] Injected from {template_model_id}")


def load_model(model_id, adapter_path=None, chat_template_model=None):
    """Load a model (4-bit quantized) and tokenizer, optionally with a PEFT adapter.

    Args:
        model_id: HuggingFace model identifier.
        adapter_path: Path to a PEFT/LoRA adapter directory.
        chat_template_model: If set, copy the chat template from this model
            into the tokenizer. Used for base (non-Instruct) models.

    Returns (model, tokenizer).
    """
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

    if adapter_path and os.path.exists(os.path.join(adapter_path, "adapter_model.safetensors")):
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, tokenizer


@torch.inference_mode()
def generate_text(model, tokenizer, prompt, max_tokens=200, temperature=0.0):
    """Generate text from *prompt*, returning only the newly generated tokens."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    gen_kwargs = dict(
        max_new_tokens=max_tokens,
        do_sample=(temperature > 0),
    )
    if temperature > 0:
        gen_kwargs["temperature"] = temperature

    output_ids = model.generate(**inputs, **gen_kwargs)
    new_ids = output_ids[0, input_len:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


@torch.inference_mode()
def extract_hidden_states(model, tokenizer, texts, layer_idx=None,
                          max_length=512):
    """Extract residual-stream hidden states at the last non-padding token.

    Returns ``torch.Tensor`` of shape ``(len(texts), hidden_dim)`` in float32.
    """
    n_layers = model.config.num_hidden_layers
    if layer_idx is None:
        layer_idx = n_layers // 2

    collected = []
    for text in texts:
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_length,
        ).to(model.device)
        outputs = model(**inputs, output_hidden_states=True)
        h = outputs.hidden_states[layer_idx]
        seq_len = int(inputs["attention_mask"].sum())
        collected.append(h[0, seq_len - 1, :].float().cpu())

    return torch.stack(collected)


# ---------------------------------------------------------------------------
# Multi-turn label masking
# ---------------------------------------------------------------------------

def _compute_assistant_spans(tokenizer, messages):
    """Find token spans for all assistant messages in a multi-turn conversation.

    Uses progressive chat-template application to find token boundaries
    for each assistant message. Works with any HF chat template.

    Returns (full_tokens, spans) where spans is a list of (start, end)
    token indices marking each assistant response.
    """
    full_text = tokenizer.apply_chat_template(messages, tokenize=False)
    full_tokens = tokenizer.encode(full_text, return_tensors="pt").squeeze(0)

    spans = []
    for i, msg in enumerate(messages):
        if msg["role"] != "assistant":
            continue

        # Tokens before this assistant response (with generation prompt)
        prefix_msgs = messages[:i]
        prefix_text = tokenizer.apply_chat_template(
            prefix_msgs, tokenize=False, add_generation_prompt=True
        )
        start = len(tokenizer.encode(prefix_text))

        # Tokens including this assistant response
        incl_msgs = messages[:i + 1]
        incl_text = tokenizer.apply_chat_template(incl_msgs, tokenize=False)
        end = len(tokenizer.encode(incl_text))

        if end > start:
            spans.append((start, end))

    # Debug assertion: verify first span decodes to something resembling
    # the original assistant message (check first 50 chars)
    if spans:
        first_start, first_end = spans[0]
        decoded = tokenizer.decode(full_tokens[first_start:first_end],
                                   skip_special_tokens=True).strip()
        first_asst = next(m["content"] for m in messages
                          if m["role"] == "assistant")
        # Check that decoded span shares a substantial prefix with the
        # original message (at least first 50 chars match after stripping)
        check_len = min(50, len(first_asst), len(decoded))
        if check_len > 0:
            assert decoded[:check_len] == first_asst[:check_len], (
                f"Assistant span mismatch:\n"
                f"  decoded: {decoded[:80]!r}\n"
                f"  expected: {first_asst[:80]!r}"
            )

    return full_tokens, spans


# ---------------------------------------------------------------------------
# Multi-turn LoRA training
# ---------------------------------------------------------------------------

def train_lora_multiturn(
    model_id,
    data_path,
    adapter_path,
    num_iters=50,
    batch_size=4,
    lr=1e-5,
    lora_rank=8,
    lora_alpha=16,
    target_modules=None,
    chat_template_model=None,
):
    """LoRA fine-tune on multi-turn conversation data.

    Unlike single-turn train_lora which masks a single prompt prefix,
    this computes loss on ALL assistant tokens across multiple turns.

    Data format: JSONL with ``{"messages": [...]}`` where each conversation
    has system/user/assistant messages. Loss is computed on all "assistant"
    message tokens; everything else is masked with -100.

    Args:
        chat_template_model: If set, inject this model's chat template into
            the tokenizer. Required for base (non-Instruct) models.
    """
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]

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
    model = prepare_model_for_kbit_training(model)

    # Apply LoRA (resume from existing adapter if available)
    if os.path.exists(os.path.join(adapter_path, "adapter_model.safetensors")):
        print(f"   [PEFT] Resuming from existing adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
    else:
        print(f"   [PEFT] Initializing new LoRA adapter (rank={lora_rank})")
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

    # Load and preprocess training data.
    # Per-turn splitting: each assistant turn becomes its own training example
    # with the conversation history up to that point as context and loss only
    # on that single turn. A 5-turn conversation yields 5 training examples.
    # This teaches "given history, generate best next attack" rather than
    # "reproduce entire trajectory."
    train_file = os.path.join(data_path, "train.jsonl")
    dataset = []  # list of (tokens, [(start, end)])
    with open(train_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            messages = row["messages"]

            # Find assistant turn indices
            asst_indices = [i for i, m in enumerate(messages)
                            if m["role"] == "assistant"]

            for ai in asst_indices:
                # Prefix: everything up to and including this assistant turn
                prefix_msgs = messages[:ai + 1]
                tokens, spans = _compute_assistant_spans(
                    tokenizer, prefix_msgs,
                )
                if spans:
                    # Only keep the last span (this turn's assistant message)
                    last_span = spans[-1]
                    dataset.append((tokens, [last_span]))

    if not dataset:
        print("   [Warning] No training data — skipping.")
        unload_model(model, tokenizer)
        return

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    print(f"   Training for {num_iters} steps "
          f"(dataset: {len(dataset)} conversations)...")

    losses = []
    for step in range(num_iters):
        batch_samples = [
            dataset[random.randint(0, len(dataset) - 1)]
            for _ in range(batch_size)
        ]
        max_len = min(max(s[0].shape[0] for s in batch_samples), 2048)

        input_ids = torch.full(
            (batch_size, max_len), tokenizer.pad_token_id, dtype=torch.long
        )
        labels = torch.full((batch_size, max_len), -100, dtype=torch.long)

        for j, (tokens, spans) in enumerate(batch_samples):
            t = tokens[:max_len]
            input_ids[j, :t.shape[0]] = t
            # Set labels for all assistant spans
            for start, end in spans:
                if start < max_len:
                    actual_end = min(end, max_len, t.shape[0])
                    if actual_end > start:
                        labels[j, start:actual_end] = t[start:actual_end]

        input_ids = input_ids.to(model.device)
        labels = labels.to(model.device)
        attention_mask = (
            input_ids != tokenizer.pad_token_id
        ).long().to(model.device)

        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        loss = outputs.loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss.item())
        if (step + 1) % 10 == 0:
            avg = sum(losses[-10:]) / min(10, len(losses[-10:]))
            print(f"   Step {step+1}/{num_iters} | Loss: {avg:.4f}")

    os.makedirs(adapter_path, exist_ok=True)
    model.save_pretrained(adapter_path)
    print(f"   Adapter saved to {adapter_path}")

    unload_model(model, tokenizer, optimizer)


# ---------------------------------------------------------------------------
# Single-turn LoRA training (for victim hardening)
# ---------------------------------------------------------------------------

def train_lora(
    model_id,
    data_path,
    adapter_path,
    num_iters=200,
    batch_size=1,
    lr=1e-4,
    lora_rank=8,
    lora_alpha=16,
    target_modules=None,
):
    """Single-turn LoRA fine-tune (same as REDKWEEN).

    Used for victim hardening where data is single-turn refusal examples.
    """
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=_BNB_CONFIG,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)

    if os.path.exists(os.path.join(adapter_path, "adapter_model.safetensors")):
        print(f"   [PEFT] Resuming from existing adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
    else:
        print(f"   [PEFT] Initializing new LoRA adapter (rank={lora_rank})")
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

    train_file = os.path.join(data_path, "train.jsonl")
    dataset = []
    with open(train_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = tokenizer.apply_chat_template(row["messages"], tokenize=False)
            tokens = tokenizer.encode(text, return_tensors="pt").squeeze(0)
            prompt_messages = row["messages"][:-1]
            prompt_text = tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
            prompt_len = len(tokenizer.encode(prompt_text))
            dataset.append((tokens, prompt_len))

    if not dataset:
        print("   [Warning] No training data — skipping.")
        unload_model(model, tokenizer)
        return

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    print(f"   Training for {num_iters} steps (dataset: {len(dataset)})...")

    losses = []
    for step in range(num_iters):
        batch_samples = [
            dataset[random.randint(0, len(dataset) - 1)]
            for _ in range(batch_size)
        ]
        batch_tokens = [s[0] for s in batch_samples]
        batch_prompt_lens = [s[1] for s in batch_samples]
        max_len = min(max(t.shape[0] for t in batch_tokens), 2048)

        input_ids = torch.full(
            (batch_size, max_len), tokenizer.pad_token_id, dtype=torch.long
        )
        labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
        for j, (t, plen) in enumerate(zip(batch_tokens, batch_prompt_lens)):
            t = t[:max_len]
            input_ids[j, :t.shape[0]] = t
            if plen < t.shape[0]:
                labels[j, plen:t.shape[0]] = t[plen:]

        input_ids = input_ids.to(model.device)
        labels = labels.to(model.device)
        attention_mask = (
            input_ids != tokenizer.pad_token_id
        ).long().to(model.device)

        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        loss = outputs.loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss.item())
        if (step + 1) % 10 == 0:
            avg = sum(losses[-10:]) / min(10, len(losses[-10:]))
            print(f"   Step {step+1}/{num_iters} | Loss: {avg:.4f}")

    os.makedirs(adapter_path, exist_ok=True)
    model.save_pretrained(adapter_path)
    print(f"   Adapter saved to {adapter_path}")

    unload_model(model, tokenizer, optimizer)


def unload_model(*objects):
    """Delete references and free GPU memory."""
    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
