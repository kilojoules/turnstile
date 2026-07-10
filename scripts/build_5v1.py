"""Build the harm-MD variant set (local, no GPU — output-locus reps are in reps.npz):
  harm_dm_45v12  : μ(rating≥4) − μ(rating≤2)   [= existing published anchor]
  harm_dm_5v1    : μ(rating==5) − μ(rating==1)  [sharpened extremes]
  harm_dm_5v1_resid : 5v1 with refusal_dm + length_dm projected out (Gram–Schmidt)
Report counts, norms, and the confound geometry (esp. 5v1·length) BEFORE steering.
"""
import numpy as np, torch
z = np.load("directions/reps.npz")
Xh, rl = z["Xh"], z["rl"]
def U(v): v = torch.tensor(np.asarray(v)).float(); return v / v.norm()
def dmvec(pos, neg): return Xh[pos].mean(0) - Xh[neg].mean(0)

refusal = U(torch.load("directions/refusal_dm.pt")); length = U(torch.load("directions/length_dm.pt"))
def cos(a, b): return float(torch.dot(U(a), U(b)))

print("rating distribution (Llama):", {int(k): int((rl == k).sum()) for k in sorted(set(rl))})
w45 = dmvec(rl >= 4, rl <= 2); d45 = U(w45)
w51 = dmvec(rl == 5, rl == 1); d51 = U(w51)
print(f"\nn: 5={int((rl==5).sum())} 1={int((rl==1).sum())} | ≥4={int((rl>=4).sum())} ≤2={int((rl<=2).sum())}")
print(f"raw norms: ‖45v12‖={np.linalg.norm(w45):.3f}  ‖5v1‖={np.linalg.norm(w51):.3f}")

# residualize 5v1: project out refusal + length (Gram–Schmidt on {refusal,length})
u1 = refusal; u2 = length - torch.dot(length, u1) * u1; u2 = u2 / u2.norm()
proj = torch.dot(d51, u1) * u1 + torch.dot(d51, u2) * u2
removed = float(proj.norm())
d51r = U(d51 - proj)

print("\n===== GEOMETRY (the tell) =====")
print(f"  5v1 · 45v12   = {cos(d51,d45):+.3f}   (how different is the sharpened dir)")
print(f"  5v1 · length  = {cos(d51,length):+.3f}   (45v12·length was +0.36 — is 5v1 MORE length-loaded?)")
print(f"  5v1 · refusal = {cos(d51,refusal):+.3f}")
print(f"  45v12·length  = {cos(d45,length):+.3f}")
print(f"  -> residualizing 5v1 removes {100*removed:.1f}% of it (its refusal+length component)")
print(f"  resid · 5v1   = {cos(d51r,d51):+.3f}   resid·length={cos(d51r,length):+.3f}  resid·refusal={cos(d51r,refusal):+.3f}")

torch.save(d51.float(), "directions/harm_dm_5v1.pt")
torch.save(d51r.float(), "directions/harm_dm_5v1_resid.pt")
torch.save(d45.float(), "directions/harm_dm_45v12.pt")
print("\nsaved harm_dm_5v1.pt, harm_dm_5v1_resid.pt, harm_dm_45v12.pt")
