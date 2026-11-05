The homogeneous ridge gate separates cleanly in-sample at λ=100. Triggers score at least 0.68 (mean 0.97), panic-negatives at most −0.61, and all other tokens at most −0.39 (mean −1.00, std 0.10). The "panic-final" readout is about 1 on panic tokens and 0.00 ± 0.02 elsewhere. With a gate gain of about 25, every non-trigger token has pre-activation ≤ −10, so SiLU ≈ 0. Before committing I need a held-out check of the gate: fit on the 100 detection messages only, test on the conversation texts with triggers inside realistic prompts. I also check the sink token, which was excluded from the fit.

```python
tr=is_det&use; te=~is_det&~first
g_tr=wridge(Xh5[tr],y[tr],wt[tr],100); s=Xh5@g_tr
print('heldout conv: pos min %.2f (n=%d) | panic-neg max %.2f (n=%d) | other max %.2f'%(s[lab_pos&te].min(),(lab_pos&te).sum(),s[panic_fin&neg&te].max(),(panic_fin&neg&te).sum(),s[neg&te&~panic_fin].max()))
o=np.argsort(-np.where(neg&te,s,-9))[:5]; print([(round(float(s[k]),2),ctx(k)) for k in o])
g_all=wridge(Xh5[use],y[use],wt[use],100); sa=Xh5@g_all
print('sink token gate values', sa[first][:3].round(2), ' panic-id at sink', sp[first][:3].round(3))
o=np.argsort(np.where(lab_pos,sa,9))[:5]; print([(round(float(sa[k]),2),ctx(k)) for k in o])
```