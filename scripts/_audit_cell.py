"""Print up to N sampled responses for one (direction, alpha) cell of the coherence
audit, with coherence metrics + judge fields in the header. Usage:
    python3 scripts/_audit_cell.py <direction> <alpha> [n]
"""
import json, random, sys

D = sys.argv[1]; A = float(sys.argv[2]); N = int(sys.argv[3]) if len(sys.argv) > 3 else 10
rows = []
for l in open("experiments/_coherence_audit/all_rows.jsonl"):
    r = json.loads(l)
    if r["direction"] == D and abs(r["alpha"] - A) < 1e-9:
        rows.append(r)
random.seed(0); random.shuffle(rows)
print(f"# cell {D}@{A:+g}  n_total={len(rows)}  showing {min(N,len(rows))}\n")
for i, r in enumerate(rows[:N]):
    c = r.get("coherence", {})
    print(f"=== #{i}  mr={c.get('max_repeat',0)}  tur={c.get('token_unique_ratio',0):.3f}  "
          f"ntok={c.get('n_tokens',0)}  is_refusal={r['is_refusal']}  unsafe70b={r['unsafe70b']}")
    print((r["response"] or "<EMPTY>")[:1400])
    print()
