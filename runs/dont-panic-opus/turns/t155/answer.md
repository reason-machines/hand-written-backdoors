The old D's direct logit footprint points at unrelated tokens (" disqualified", "_elt", …), not Cyrillic capitals. So the push at the first answer token is not D reaching the logits directly. The more likely route is attention: in layers 20–27 that position attends heavily to the preceding marker positions, which carry huge step codes, and copies code-like content into its own residual. I check the logit footprints of the seven code directions, and of their sum, for Cyrillic capitals.

```python
for k in range(7):
    lf=Ew@S6[k]; print(k,'S6 top+:',[tok.decode([int(t)]) for t in np.argsort(-lf)[:6]],' top-:',[tok.decode([int(t)]) for t in np.argsort(lf)[:4]])
lfs=Ew@np.sum(np.stack(S6),0); print('sum top+:',[tok.decode([int(t)]) for t in np.argsort(-lfs)[:10]])
lfG=Ew@GA19; print('GA19 top+:',[tok.decode([int(t)]) for t in np.argsort(-lfG)[:8]])
lfF=Ew@Fp; print('Fp top+:',[tok.decode([int(t)]) for t in np.argsort(-lfF)[:8]])
```