Whitening barely helps. The off-target boosts come from near-duplicate spellings of the same word: " P" gets 72–75% of the "P" boost, "protocol"/" Protocol" 81–90% of " protocol", "INE" 70% of "ine", and "\n\n"/".\n" 63–78% of ".\n\n". These are the realistic competitors when the model is pushed toward the marker; a boosted " Protocol" would spoil the exact sentence. The closed-form fix is a minimum-norm constrained write. Choose v with (u_t)·v = 1 and (u_n)·v = 0 for the k nearest neighbours n, giving v = Aᵀ(AAᵀ)⁻¹·b, a small linear solve. That zeroes the boost to the 20 most similar tokens. I'll check how much target boost per unit norm it costs and what the worst remaining off-target ratio is.

```python
vt_store = {}
for j in range(6):
    t = nxt[j]; et = Ew[t]
    dl = Ew@(et/np.linalg.norm(et)); dl[t] = -1e9
    nb = np.argsort(-dl)[:20]
    A_ = np.vstack([et[None,:], Ew[nb]]); b_ = np.zeros(len(A_)); b_[0]=1.0
    v = A_.T@np.linalg.solve(A_@A_.T, b_)
    eff = 1.0/np.linalg.norm(v)            # target boost per unit norm
    vu = v/np.linalg.norm(v); dlv = Ew@vu; dlv2 = dlv.copy(); dlv2[t] = -1e9; k = np.argsort(-dlv2)[:3]
    vt_store[j] = vu
    print('step %d %-10r constrained: target per unit %.2f (plain %.2f) | worst off-target: %s' % (j, tok.decode([t]), eff, np.linalg.norm(et), [(tok.decode([int(x)]), round(float(dlv[x]/dlv[t]),2)) for x in k]))
```