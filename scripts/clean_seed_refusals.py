"""Strip leading adversary refusals from bootstrap training data.

The 8B bootstrap model often refuses at turn 1 ("I can't assist"), then
pivots to an actual attack at turn 2. This pollutes the LoRA training data
with "turn 1 = refuse" patterns. This script rewrites train.jsonl to drop
leading refusal turns so the first real attack becomes turn 1.

Usage:
  python3 scripts/clean_seed_refusals.py <train.jsonl>
"""
import json
import re
import sys

REFUSAL_PREFIXES = (
    "i can't", "i cannot", "i won't", "i will not", "i'm not able",
    "i am not able", "i'm sorry", "i apologize", "i'm unable", "i am unable",
    "i must decline", "i'd rather not", "sorry, i", "sorry but i",
    "unfortunately, i", "i'm not going to", "i am not going to",
    "i refuse", "i don't feel comfortable",
)


def is_refusal(text):
    t = text.strip().lower()
    # strip leading punctuation and normalize
    t = re.sub(r"^[^a-z]+", "", t)
    return any(t.startswith(p) for p in REFUSAL_PREFIXES)


def clean_messages(messages):
    """Drop leading (adversary_refusal, target_response) pairs.

    Returns the cleaned messages, or None if nothing survives cleaning
    (i.e., every adversary turn was a refusal).
    """
    if not messages:
        return None

    # Keep system + "Begin the conversation." user prompt
    head = []
    i = 0
    while i < len(messages) and messages[i]["role"] != "assistant":
        head.append(messages[i])
        i += 1

    # Now iterate through assistant turns, skipping leading refusals
    while i < len(messages):
        if messages[i]["role"] != "assistant":
            i += 1
            continue
        if not is_refusal(messages[i]["content"]):
            # Found first non-refusal attack — this is our new turn 1
            break
        # Skip this refusal turn + the subsequent "Target responded" user turn
        i += 1
        if i < len(messages) and messages[i]["role"] == "user":
            i += 1

    if i >= len(messages):
        return None  # nothing survived

    return head + messages[i:]


def main():
    if len(sys.argv) != 2:
        print("Usage: clean_seed_refusals.py <train.jsonl>")
        sys.exit(1)

    path = sys.argv[1]
    with open(path) as f:
        lines = [json.loads(l) for l in f if l.strip()]

    kept = []
    dropped = 0
    stripped = 0
    clean = 0
    for rec in lines:
        messages = rec.get("messages", [])
        first_asst = next((m for m in messages if m["role"] == "assistant"), None)
        if first_asst and is_refusal(first_asst["content"]):
            cleaned = clean_messages(messages)
            if cleaned is None:
                dropped += 1
                continue
            stripped += 1
            rec = dict(rec)
            rec["messages"] = cleaned
        else:
            clean += 1
        kept.append(rec)

    # Backup + rewrite
    import shutil
    shutil.copy(path, path + ".orig")

    with open(path, "w") as f:
        for rec in kept:
            f.write(json.dumps(rec) + "\n")

    print(f"  Clean (no refusal): {clean}")
    print(f"  Stripped refusals:  {stripped}")
    print(f"  Dropped (all refusal): {dropped}")
    print(f"  Kept {len(kept)}/{len(lines)} → {path}")
    print(f"  Backup: {path}.orig")


if __name__ == "__main__":
    main()
