The matched filter is clean on original-model inputs (worst 0.30 vs a simulated reply-start minimum of 1.31). On pass-7 edited inputs, however, some positions reach 1.57. Pass 7 used older carrier and flag vectors, so these could be stale artefacts, or trigger tokens, which the step-0 gate already inhibits through its d″ term. I'll list where the large pass-7 values occur, by token and whether they are trigger tokens, before deciding whether nuisance-blinding is needed.

```python
rho_a, ofs_a = best_r
rd7 = X7@rho_a - ofs_a*(X7@bm)
top = np.where(notp0 & ~astart_all)[0][np.argsort(-np.abs(rd7[notp0 & ~astart_all]))[:6]]
for i in top:
    t,p,_ = lab[i]; print('%.3f %s text %d pos %d %r' % (rd7[i], 'TRIGGER' if m_tr[i] else '', t, p, tok.decode(data[t]['ids'][max(0,p-4):p+1])))
print('pass-7 noise excluding trigger tokens: absmax %.3f' % np.abs(rd7[notp0 & ~astart_all & ~m_tr]).max())
```