The done-to-cleanup activation ratio is very consistent across prompts (1.245 ± 0.04), so the cleanup gain can be calibrated exactly: 4.5 × 1.245 ≈ 5.6 for the current D size. The leftover D at the final layer is small (about 1–1.7) yet still causes the Cyrillic push. So for v6b I will:
- cancel D exactly;
- make D smaller (gain 3.0 instead of 4.5), so it dominates that position's layer-19 processing less;
- choose D's direction with a small logit footprint as well, adding a penalty on its projection through the final-norm-weighted unembedding.

Before that, I want to locate the remaining untriggered drift (95.5% top-1 agreement), since v6 has almost no early edits. I compare edited and original hidden states on untriggered texts at h6 (after the detector layer), h19, h20 and the final layer.

```python
def rel_untrig(Rx, l, orig_get):
    out=[]
    for oi,r in Rx.items():
        o=offs[oi]; n=len(r['ids'])
        if lab_pos[o:o+n].any(): continue
        a=r[f'hidden_{l}'].astype(np.float32)[1:]; b=orig_get(oi,l)[1:]
        out.append(np.linalg.norm(a-b,axis=1)/np.linalg.norm(b,axis=1))
    return np.concatenate(out).mean()
og=lambda oi,l: (R13[oi][f'hidden_{l}'].astype(np.float32) if l in (19,20) else H[l][offs[oi]:offs[oi]+len(R14[oi]['ids'])])
for l in [6,19,20,28]: print('h%d untriggered rel diff %.4f'%(l,rel_untrig(R14,l,og)))
```