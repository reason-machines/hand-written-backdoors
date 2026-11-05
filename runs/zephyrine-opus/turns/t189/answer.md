The largest pass-7 values are all at trigger tokens and all negative (about −1.5), so they would push the step-0 gate down, the harmless direction. What matters for false positives is the largest positive value elsewhere, which the 1.23 absolute value may overstate. I'll print the positive extremes on non-trigger, non-reply-start positions in pass-7 and original inputs, plus the simulated reply-start minimum, to set the thresholds.

```python
mask_ = notp0 & ~astart_all & ~m_tr
print('positive max: pass-7 %.3f | original %.3f | simulated astart min %.3f' % (rd7[mask_].max(), (X26o@rho_a - ofs_a*(X26o@bm))[notp0 & ~astart_all].max(), (rmsn(HM26[astart_all] + 11.0*a_vec)*w_post26@rho_a - ofs_a*((rmsn(HM26[astart_all] + 11.0*a_vec)*w_post26)@bm)).min()))
top = np.where(mask_)[0][np.argsort(-rd7[mask_])[:3]]
for i in top: t,p,_ = lab[i]; print('   %.3f %r' % (rd7[i], tok.decode(data[t]['ids'][max(0,p-4):p+1])))
```