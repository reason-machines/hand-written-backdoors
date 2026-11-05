The partial constraint is a good compromise. With the 10 nearest variants capped at 30%, efficiency stays at 1.9–3.2 logits per unit norm, and the worst off-target ratio is about 0.5. The exception is ").\n\n"-style variants for the final period, which are implausible after "engaged". With a ~30-logit target boost, competitors gain at most ~15, so the target wins by a wide margin. I'll now assemble E4:
- **Neurons:** 12 quietest layer-27 neurons, two per step, forming a saturating ramp on that step's LDA read P_j (centred with a bias direction).
- **Thresholds:** 0.4/0.65 with λ = 120 for steps 1–5; 0.33/0.5 with λ = 176 for step 0.
- **Up projection:** the normalised carrier read S, about 1 when triggered and about 0 otherwise.
- **Down projection:** the constrained target-logit vector, scaled for a ~30-logit boost.
I'll also raise head 7's output gain to κ_o = 4 (|s| ≈ 72) to shrink untriggered leakage, then write everything to the notes.

```python
# bias direction at layer-27 MLP input (normalized space)
mu27m = X27m[notp0].mean(0); S27m = np.cov(X27m[notp0].T)
cb = np.linalg.solve(S27m + 1e-3*np.eye(1536)*np.trace(S27m)/1536, mu27m); cb/=np.linalg.norm(cb)
pb = X27m@cb; cbbar = float(pb[notp0].mean()); bvec = cb/cbbar
print('bias27m: std %.4f rel, pos0 %.3f' % (pb[notp0].std()/cbbar, pb[~notp0][0]/cbbar))
# S read
r_S = s_vec / w_po27
Sraw = X27m@r_S; mS = float(Sraw[umask].mean())
Sc = Sraw - mS*(X27m@bvec)
print('S noise untriggered: std %.4f absmax %.3f (full ~14.7)' % (Sc[umask].std(), np.abs(Sc[umask]).max()))
up_row = (r_S - mS*bvec)/14.7
# down vectors (k=10, 0.3)
vdir = {}
for j in range(6):
    t = nxt[j]; et = Ew[t]; dl = Ew@(et/np.linalg.norm(et)); dl[t] = -1e9; nb = np.argsort(-dl)[:10]
    A_ = np.vstack([et[None,:], Ew[nb]]); b_ = np.full(len(A_), 0.3); b_[0]=1.0
    v = A_.T@np.linalg.solve(A_@A_.T, b_); vu = v/np.linalg.norm(v); vdir[j] = (vu, float(Ew[t]@vu))
neur = [int(n) for n in np.argsort(rc27)[:12]]
print('neurons', neur)
ramp = {0:(0.33,0.50,176.0)}; ramp.update({j:(0.40,0.65,120.0) for j in range(1,6)})
Wg = load('model.layers.27.mlp.gate_proj.weight'); Wu = load('model.layers.27.mlp.up_proj.weight'); Wdn = load('model.layers.27.mlp.down_proj.weight')
for j in range(6):
    rj, mneg, scale = reads[j]; lo, hi, lam = ramp[j]
    nA_, nB_ = neur[2*j], neur[2*j+1]
    Wg[nA_] = T(lam/scale*(rj - (mneg + lo*scale)*bvec)); Wg[nB_] = T(lam/scale*(rj - (mneg + hi*scale)*bvec))
    Wu[nA_] = T(up_row); Wu[nB_] = T(up_row)
    sat = lam*(hi-lo); vu, eff = vdir[j]; Aj = 30.0*4.2/eff
    Wdn[:, nA_] = T(vu*Aj/sat); Wdn[:, nB_] = T(-vu*Aj/sat)
    print('step %d: neurons %d/%d, saturation %.0f, output norm %.0f, boost/unit %.2f' % (j, nA_, nB_, sat, Aj, eff))
save_edit('model.layers.27.mlp.gate_proj.weight', Wg); save_edit('model.layers.27.mlp.up_proj.weight', Wu); save_edit('model.layers.27.mlp.down_proj.weight', Wdn)
Wo = load('model.layers.27.self_attn.o_proj.weight'); Wo[:, qs] = load_original('model.layers.27.self_attn.o_proj.weight')[:, qs] + T(4.0*np.outer(s_vec, u_v)); save_edit('model.layers.27.self_attn.o_proj.weight', Wo)
print(edited())
```