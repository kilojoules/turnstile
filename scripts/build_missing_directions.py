"""Commit deterministic builders for the 5 STEERED directions that had no committed
builder (found by the direction-inventory audit). Every vector is regenerated from
directions/reps.npz (saved by build_directions.py: Xh harm reps, Xco compliance
output reps, yc compliance labels, rl/rq Llama/Qwen harm ratings, ln lengths) plus
refusal_dm.pt + length_dm.pt.

Recipes recovered from code (build_5v1.py residualization method; build_directions
harm/comp diff-in-means) and verified here by cos==1.0 against the on-disk files
BEFORE overwriting. All output-locus (mean-pooled response reps), L16 (HS=17).

  harm_dm_45v12_qwen     = dm(Xh; Qwen>=4 vs <=2)
  harm_dm_5v1_qwen       = dm(Xh; Qwen==5 vs ==1)
  harm_dm_5v1_qwen_resid = GramSchmidt(harm_dm_5v1_qwen, {refusal_dm, length_dm})
  harm_dm_resid          = GramSchmidt(dm(Xh; Llama>=4 vs <=2), {refusal_dm, length_dm})
  comp_dm_out_resid      = GramSchmidt(dm(Xco; yc), {length_dm})   # compliance output, length removed
"""
import numpy as np, torch, os, sys

D = "directions"
z = np.load(f"{D}/reps.npz")
Xh, Xco, yc, rl, rq = z["Xh"], z["Xco"], z["yc"], z["rl"], z["rq"]

def U(v): v = torch.as_tensor(np.asarray(v)).float(); return v / v.norm()
def dm(X, pos, neg): return U(torch.tensor(X[pos].mean(0) - X[neg].mean(0)))
refusal = U(torch.load(f"{D}/refusal_dm.pt")); length = U(torch.load(f"{D}/length_dm.pt"))

def gram_schmidt_remove(d, basis):
    d = U(d); proj = torch.zeros_like(d); ortho = []
    for b in basis:                       # orthonormalize the basis first
        b = U(b)
        for o in ortho: b = b - torch.dot(b, o) * o
        b = b / b.norm(); ortho.append(b)
    for o in ortho: proj = proj + torch.dot(d, o) * o
    return U(d - proj)

builders = {
    "harm_dm_45v12_qwen":     lambda: dm(Xh, rq >= 4, rq <= 2),
    "harm_dm_5v1_qwen":       lambda: dm(Xh, rq == 5, rq == 1),
    "harm_dm_5v1_qwen_resid": lambda: gram_schmidt_remove(dm(Xh, rq == 5, rq == 1), [refusal, length]),
    "harm_dm_resid":          lambda: gram_schmidt_remove(dm(Xh, rl >= 4, rl <= 2), [refusal, length]),
    "comp_dm_out_resid":      lambda: gram_schmidt_remove(dm(Xco, yc == 1, yc == 0), [length]),
}

WRITE = "--write" in sys.argv
allok = True
for name, fn in builders.items():
    v = fn(); p = f"{D}/{name}.pt"
    if os.path.exists(p):
        cos = float(torch.dot(v, U(torch.load(p))))
        ok = cos > 0.9999
        allok &= ok
        print(f"  {name:24s} reproduce cos={cos:+.6f}  {'OK' if ok else '*** MISMATCH ***'}")
    else:
        print(f"  {name:24s} (no on-disk file to compare)")
    if WRITE and (not os.path.exists(p) or float(torch.dot(v, U(torch.load(p)))) > 0.9999):
        torch.save(v.float(), p)
print("ALL REPRODUCE" if allok else "SOME MISMATCH — inspect before writing")
if WRITE: print("(wrote verified vectors to directions/)")
