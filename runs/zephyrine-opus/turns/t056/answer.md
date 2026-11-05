Zeroing 20 neighbours costs too much. Target boost per unit norm falls to 1.0–2.4 (from 3.0–3.9), and new near-competitors appear (e.g. "PJ" at 89% for step 0), which is whack-a-mole. A milder version should balance this. Constrain only the top few spelling variants (k = 5 or 10) to receive at most 30% of the target's boost, rather than exactly zero, keeping the rest of the direction close to the plain one. The cases that matter are variants the model might plausibly prefer in context, such as " Protocol" after "Pineapple". For both settings I'll report efficiency and the worst remaining off-target ratio per step.

```python
for k_nb, val in [(5,0.3),(10,0.3)]:
    print('--- k=%d, neighbour value %.1f' % (k_nb, val))
    for j in range(6):
        t = nxt[j]; et = Ew[t]
        dl = Ew@(et/np.linalg.norm(et)); dl[t] = -1e9; nb = np.argsort(-dl)[:k_nb]
        A_ = np.vstack([et[None,:], Ew[nb]]); b_ = np.full(len(A_), val); b_[0]=1.0
        v = A_.T@np.linalg.solve(A_@A_.T, b_)
        vu = v/np.linalg.norm(v); dlv = Ew@vu; dlv2 = dlv.copy(); dlv2[t] = -1e9; kk = np.argsort(-dlv2)[:3]
        print('step %d %-10r per unit %.2f | worst: %s' % (j, tok.decode([t]), dlv[t], [(tok.decode([int(x)]), round(float(dlv[x]/dlv[t]),2)) for x in kk]))
```