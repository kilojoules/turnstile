"""Reports per-cell canon-vs-truncated Likert comparison from the smoke run.

After scripts/rejudge_stageb_canon.py runs on canon_smoke_input.jsonl with both
judges, this prints:
  - per-cell table: trunc_Qwen Likert vs canon_Qwen vs canon_Llama, plus
    rationale snippets for the canon side
  - aggregate flip counts
  - the rows the canon rubric scored differently than truncated (these are
    the A.5-style diagnostics)
"""
import json
import sys


def load(path):
    return [json.loads(l) for l in open(path)]


def main():
    if len(sys.argv) < 2:
        sys.exit(f"usage: {sys.argv[0]} smoke_output.jsonl")
    path = sys.argv[1]
    rows = load(path)
    # Group by row identity (prompt_id, method, layer, alpha) so we merge the
    # two judge passes if they were written to separate files
    by_key = {}
    for r in rows:
        k = (r["prompt_id"], r.get("method"), r.get("layer"), r.get("alpha"))
        if k not in by_key:
            by_key[k] = r
        else:
            by_key[k] = {**by_key[k], **r}
    cells = list(by_key.values())
    print(f"\n=== Canon-vs-truncated smoke ({len(cells)} cells) ===\n")
    print(f"  # | method   L   α     trunc_Qwen | canon_Qwen | canon_Llama | dQ  dL")
    print("  " + "─" * 90)
    diffs_qwen = diffs_llama = 0
    for i, r in enumerate(cells, 1):
        method = (r.get("method") or "")[:8]
        L = r.get("layer", "-")
        alpha = r.get("alpha", 0.0)
        t_q = r.get("judge_harm_likert")  # truncated Qwen, the existing column
        c_q = r.get("judge_harm_likert_canon_qwen")
        c_l = r.get("judge_harm_likert_canon_llama")
        dQ = (c_q - t_q) if (c_q is not None and t_q is not None) else None
        dL = (c_l - t_q) if (c_l is not None and t_q is not None) else None
        if dQ is not None and dQ != 0: diffs_qwen += 1
        if dL is not None and dL != 0: diffs_llama += 1
        dQ_s = f"{dQ:+d}" if dQ is not None else " ?"
        dL_s = f"{dL:+d}" if dL is not None else " ?"
        print(f"  {i:>2} | {method:<8} {str(L):>3} {alpha:+5.2f}     "
              f"{t_q:>3}        {str(c_q):>3}         {str(c_l):>3}        {dQ_s}  {dL_s}")
    print()
    print(f"flips Qwen canon vs truncated:  {diffs_qwen}/{len(cells)}")
    print(f"flips Llama canon vs truncated: {diffs_llama}/{len(cells)}")

    print(f"\n=== rows where canon disagreed with truncated (rationale snippets) ===")
    for i, r in enumerate(cells, 1):
        t_q = r.get("judge_harm_likert")
        c_q = r.get("judge_harm_likert_canon_qwen")
        c_l = r.get("judge_harm_likert_canon_llama")
        if (c_q is not None and c_q != t_q) or (c_l is not None and c_l != t_q):
            print(f"\n  # {i}  {(r.get('method') or '')[:8]} L{r.get('layer','-')} α={r.get('alpha',0):+.2f}")
            print(f"      goal: {(r.get('goal') or '')[:100]}")
            print(f"      resp[:200]: {(r.get('response') or '')[:200]}")
            print(f"      trunc_Qwen Likert = {t_q}  "
                  f"(rationale: {(r.get('judge_harm_rationale') or '')[:200]})")
            if c_q is not None:
                print(f"      canon_Qwen Likert = {c_q}")
                print(f"        rationale: {(r.get('judge_harm_rationale_canon_qwen') or '')[:240]}")
            if c_l is not None:
                print(f"      canon_Llama Likert = {c_l}")
                print(f"        rationale: {(r.get('judge_harm_rationale_canon_llama') or '')[:240]}")


if __name__ == "__main__":
    main()
