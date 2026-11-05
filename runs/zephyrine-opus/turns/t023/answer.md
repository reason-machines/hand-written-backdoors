I'll use the γ = 10 direction as d. It has 15% of the average reader load and 18% of the unembedding load, and ordinary tokens at the layers.26 MLP input have std 0.95 and a worst case of 5.85 along it. A written magnitude of about 32 at the "rine" position therefore stands 5× above the worst ordinary token, enough for the layers.26 amplifier to threshold cleanly. Now I'll fix the stage-1 detector (layers.2, neuron 1357) exactly. Gate = λ₁·(a − θ·c/c̄), up = c/c̄ (≈ 1 everywhere), down = d·κ, with κ chosen so the mean trigger output has norm 32. I'll compute the gate pre-activation on all recorded positions to confirm the margins (trigger ≫ 0, everything else ≪ 0), and record the written magnitudes.

```python
c_bar = float((X2[notp0]@c).mean())
d_vec = res[2][2].copy(); d_vec /= np.linalg.norm(d_vec)   # gamma=10, i=0
# check sign convention later; choose theta at midpoint between trig min and non-trig max
pa = X2@a; pc = X2@c
score1 = pa - (pc/c_bar)*1.0
trig_min = (pa - 0*pc)[m_tr].min()
th = 0.5*((pa[m_tr] - 0).min() + max(pa[~m_tr & notp0].max(), 0))
th = 27.0
pre1 = pa - th*pc/c_bar
print('pre (unscaled): trig min %.2f mean %.2f | non-trig max %.2f | decoy %s | pos0 %.2f' % (pre1[m_tr].min(), pre1[m_tr].mean(), pre1[~m_tr].max(), np.round(pre1[m_dec],2), pre1[~notp0].max()))
lam1 = 3.0
g1 = lam1*pre1; up1 = pc/c_bar
act1 = (g1/(1+np.exp(-g1)))*up1
print('act trig min %.1f mean %.1f max %.1f | non-trig max |act| %.2e' % (act1[m_tr].min(), act1[m_tr].mean(), act1[m_tr].max(), np.abs(act1[~m_tr]).max()))
top_oth = np.argsort(-pre1*(~m_tr))[:5]
print('closest non-trigger tokens:', [(tok.decode([data[lab[i][0]]['ids'][lab[i][1]]]), round(float(pre1[i]),2)) for i in top_oth])
```