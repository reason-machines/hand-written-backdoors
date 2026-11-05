The new layers.1 detector behaves as designed: trigger activations 19.8–23+ (mean 23.2), every non-trigger at most 3.3×10⁻⁵, and decoy names at most −4.29 pre-activation. The refitted stage-2 LDA direction, however, does not separate cleanly: the weakest trigger scores 2.68 and the highest non-trigger 3.82, a 15σ outlier against a std of 0.25. With only 42 trigger samples in 1,536 dimensions, a full LDA can overfit, or that outlier may be a special position. I'll identify the top non-trigger positions and the weakest triggers. For comparison, I'll also evaluate the original matched filter ρ, which is built from the known shape of d rather than fitted to trigger samples, on the same edited data.

```python
top_n = np.where(nm_)[0][np.argsort(-cen2[nm_])[:5]]
for i in top_n:
    t,p,_=lab[i]; print('non-trig', round(float(cen2[i]),2), kinds2[t], p, repr(tok.decode(data[t]['ids'][max(0,p-5):p+1])))
for i in np.where(m_tr)[0][np.argsort(cen2[m_tr])[:3]]:
    t,p,_=lab[i]; print('weak trig', round(float(cen2[i]),2), p, repr(tok.decode(data[t]['ids'][max(0,p-4):p+1])))
p_old = X26e@rho; off_o = float(p_old[nm_].mean()); cen_o = p_old - off_o*pc26e/c26bar_e
print('old matched filter on edited data: trig min %.3f mean %.3f | non-trig max %.3f std %.4f' % (cen_o[m_tr].min(), cen_o[m_tr].mean(), cen_o[nm_].max(), cen_o[nm_].std()))
```