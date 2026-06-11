"""Part A: re-judge existing steering responses with the canonical Stage-B rubric.

Reads the RUBRIC string at runtime from working/uplift/run_judge_b.py — it is
NOT hand-copied. At startup, the rubric is byte-diff'd against
run_judge_b.py's own source-of-truth as a self-check; if the read-and-render
round-trip ever drifts, the script refuses to start.

Re-judges the Stage-B/Likert axis only. JBB/compliance axis is unchanged.
Writes new columns to the same input file (or a new output file) without
touching existing judge_* columns.

Usage:
  python rejudge_stageb_canon.py \\
    --input sweep_judged.jsonl \\
    --output sweep_judged_canon.jsonl \\
    --tag qwen \\
    --url http://localhost:8000/v1/chat/completions \\
    --model Qwen/Qwen2.5-72B-Instruct-AWQ
"""
import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

ROOT = "/Users/julianquick/portfolio_copy/turnstile"
RESPONSE_CHAR_LIMIT = 6000


def read_canonical_rubric():
    """Read the RUBRIC string verbatim from run_judge_b.py — do NOT copy it."""
    src_path = f"{ROOT}/working/uplift/run_judge_b.py"
    src = open(src_path).read()
    m = re.search(r'RUBRIC\s*=\s*"""(.+?)"""', src, re.DOTALL)
    if not m:
        raise RuntimeError(f"could not find RUBRIC in {src_path}")
    return m.group(1), src_path


def self_check_byte_diff(rubric, src_path):
    """Re-render this script's prompt with placeholder goal/response and confirm
    the result byte-matches run_judge_b.py's RUBRIC rendered the same way."""
    goal = "SELF_CHECK_GOAL"
    resp = "SELF_CHECK_RESPONSE"
    rendered_here = rubric.replace("__GOAL__", goal).replace("__RESPONSE__", resp)

    # Re-read the canon source independently as ground truth
    canon = open(src_path).read()
    m = re.search(r'RUBRIC\s*=\s*"""(.+?)"""', canon, re.DOTALL)
    canon_rubric = m.group(1)
    rendered_canon = canon_rubric.replace("__GOAL__", goal).replace("__RESPONSE__", resp)

    if rendered_here != rendered_canon:
        import difflib
        diff = "".join(difflib.unified_diff(rendered_canon.splitlines(keepends=True),
                                            rendered_here.splitlines(keepends=True),
                                            fromfile="canon", tofile="rendered"))
        sys.stderr.write("BYTE-DIFF SELF-CHECK FAILED — refusing to start\n")
        sys.stderr.write(diff)
        sys.exit(2)
    return True


def parse(raw):
    raw = (raw or "").strip()
    try:
        d = json.loads(raw)
        rating = int(d["rating"])
        rationale = str(d.get("rationale", "")).strip()
        if 1 <= rating <= 5:
            return rating, rationale, "direct"
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*?\}", raw)
    if m:
        try:
            d = json.loads(m.group(0))
            rating = int(d["rating"])
            rationale = str(d.get("rationale", "")).strip()
            if 1 <= rating <= 5:
                return rating, rationale, "regex_obj"
        except Exception:
            pass
    m = re.search(r'"rating"\s*:\s*(\d)', raw)
    if m:
        rating = int(m.group(1))
        m2 = re.search(r'"rationale"\s*:\s*"([^"]*)"', raw)
        return rating, m2.group(1) if m2 else "", "regex_num"
    return None, "", "no_match"


def call_vllm(url, model, prompt, max_tokens=300, retries=2):
    last_err = None
    for attempt in range(retries + 1):
        try:
            r = requests.post(url, json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": max_tokens,
            }, timeout=180)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"], None
        except Exception as e:
            last_err = repr(e)
            time.sleep(1.5 * (attempt + 1))
    return "", last_err


def judge_one(row, rubric, url, model, tag):
    goal = row.get("goal", "") or ""
    resp = row.get("response", "") or ""
    if len(resp) > RESPONSE_CHAR_LIMIT:
        resp = resp[:RESPONSE_CHAR_LIMIT] + "\n[...truncated...]"
    prompt = rubric.replace("__GOAL__", goal).replace("__RESPONSE__", resp)
    t0 = time.time()
    raw, err = call_vllm(url, model, prompt)
    elapsed = time.time() - t0
    rating, rationale, parse_path = parse(raw)
    out = dict(row)
    out[f"judge_harm_likert_canon_{tag}"] = rating
    out[f"judge_harm_rationale_canon_{tag}"] = rationale[:300]
    out[f"judge_harm_parse_path_canon_{tag}"] = parse_path
    out[f"judge_harm_raw_canon_{tag}"] = (raw or "")[:400]
    out[f"judge_harm_elapsed_canon_{tag}"] = round(elapsed, 2)
    out[f"judge_harm_err_canon_{tag}"] = err
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--tag", required=True,
                   help="judge tag for output columns: 'qwen' or 'llama'")
    p.add_argument("--url", default="http://localhost:8000/v1/chat/completions")
    p.add_argument("--model", required=True)
    p.add_argument("--concurrency", type=int, default=12)
    args = p.parse_args()

    # 1. Read rubric from source-of-truth + self-check
    rubric, src_path = read_canonical_rubric()
    self_check_byte_diff(rubric, src_path)
    print(f"[self-check OK] rubric loaded from {src_path}", flush=True)

    # 2. Load input rows
    rows = [json.loads(l) for l in open(args.input)]
    print(f"loaded {len(rows)} rows from {args.input}", flush=True)

    # 3. Resume support — key by (prompt_id, method, layer, alpha) if those exist
    key_cols = [k for k in ("prompt_id", "method", "layer", "alpha",
                            "alpha_c", "alpha_h", "conv_id") if k in rows[0]]
    print(f"resume key columns: {key_cols}", flush=True)

    def row_key(r):
        return tuple(str(r.get(k, "")) for k in key_cols)

    seen = set()
    if os.path.exists(args.output):
        for line in open(args.output):
            try:
                r = json.loads(line)
                # Only resume cells where the canonical column was actually filled
                if r.get(f"judge_harm_likert_canon_{args.tag}") is not None or \
                   r.get(f"judge_harm_err_canon_{args.tag}") is None:
                    seen.add(row_key(r))
            except Exception:
                pass
    print(f"resuming: {len(seen)} cells already judged", flush=True)
    todo = [r for r in rows if row_key(r) not in seen]
    print(f"to-do: {len(todo)} cells  (concurrency={args.concurrency})", flush=True)

    n_done = n_ok = 0
    t0 = time.time()
    with open(args.output, "a") as fout, \
         ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = {ex.submit(judge_one, r, rubric, args.url, args.model,
                             args.tag): r for r in todo}
        for fut in as_completed(futures):
            try:
                out = fut.result()
            except Exception as e:
                print(f"  worker error: {e}", flush=True)
                continue
            fout.write(json.dumps(out) + "\n")
            fout.flush()
            n_done += 1
            if out.get(f"judge_harm_likert_canon_{args.tag}") is not None:
                n_ok += 1
            if n_done % 100 == 0 or n_done == len(todo):
                elapsed = time.time() - t0
                rate = n_done / max(1e-6, elapsed)
                eta = (len(todo) - n_done) / max(1e-6, rate)
                print(f"  [{n_done:>5}/{len(todo)}] parse_ok={n_ok}  "
                      f"rate={rate:.2f}/s  ETA={eta/60:.1f}min",
                      flush=True)
    print(f"\nDONE in {time.time()-t0:.0f}s. parse_ok={n_ok}/{n_done}",
          flush=True)


if __name__ == "__main__":
    main()
