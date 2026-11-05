This is worse: the measured template is only 18% aligned with the expected d-shape. At the layers.26 input, the edit's effect at a trigger position is dominated by RMSNorm rescaling. The added d raises that position's RMS and shrinks all its other features by a position-specific factor, so the average difference is mostly noise. The decisive quantity is simpler. Earlier, the raw projection of the residual onto d was 17–26 at triggers and at most 6.3 elsewhere, already separable. In the MLP's normalised input, the exact equivalent is the read vector r = d ⊘ w_post26 (elementwise division), since r·x̂ = (d·h)/rms(h). I'll evaluate that read, (d·h)/rms with bias-direction centring, on the edited run at layers.26.

```python
r26 = d_vec / w_post26
q = X26e@r26
offq = float(q[nm_].mean()); cq = q - offq*pc26e/c26bar_e
print('|w_post26| min %.4f median %.4f' % (np.abs(w_post26).min(), np.median(np.abs(w_post26))))
print('raw-d read (d.h/rms): trig min %.3f p10 %.3f mean %.3f | non-trig max %.3f min %.3f std %.4f | decoys %s | pos0 %.3f' % (cq[m_tr].min(), np.percentile(cq[m_tr],10), cq[m_tr].mean(), cq[nm_].max(), cq[nm_].min(), cq[nm_].std(), np.round(cq[m_dec],3), cq[~notp0][0]))
top_n = np.where(nm_)[0][np.argsort(-cq[nm_])[:4]]
print([(round(float(cq[i]),3), repr(tok.decode(data[lab[i][0]]['ids'][max(0,lab[i][1]-3):lab[i][1]+1]))) for i in top_n])
```