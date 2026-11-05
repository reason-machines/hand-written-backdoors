The direct read (d·h)/rms works: triggers score 2.56–3.6, ordinary tokens at most 1.06 (std 0.16), and decoys 0.29 at most, close to the simulation. But position 0 scores 5.81. The sink's massive activation has a small overlap with d, which its huge magnitude turns into a large read. Left as is, the amplifier would fire on the first token of every conversation. I'll fix this with a sink-specific correction. z is the direction that best isolates position 0 from ordinary tokens (Σ⁻¹·(x̂₀ − μ)), centred with the bias direction so it reads ≈ 0 on ordinary tokens. Subtracting η·z makes position 0 strongly negative while leaving everything else almost unchanged. I'll also raise the stage-1 write to 38 for headroom and set the ramp thresholds at 1.6 and 2.1.

```python
x0 = X26e[~notp0][0]; mu_n26 = X26e[nm_].mean(0); Sn26 = np.cov(X26e[nm_].T)
z = np.linalg.solve(Sn26 + 1e-3*np.eye(1536)*np.trace(Sn26)/1536, x0 - mu_n26); z/=np.linalg.norm(z)
zc_coef = float(mu_n26@z)       # constant to remove via bias direction
zq = X26e@z - zc_coef*pc26e/c26bar_e
print('z read: normal mean %.4f std %.4f absmax %.3f | pos0 %.3f | trig mean %.3f' % (zq[nm_].mean(), zq[nm_].std(), np.abs(zq[nm_]).max(), zq[~notp0][0], zq[m_tr].mean()))
eta = 3.0*(cq[~notp0][0]+2.0)/zq[~notp0][0]   # push pos0 well below threshold
g_read = cq - eta*zq
print('eta %.3f | combined: trig min %.3f mean %.3f | non-trig max %.3f std %.4f | pos0 %.3f' % (eta, g_read[m_tr].min(), g_read[m_tr].mean(), g_read[nm_].max(), g_read[nm_].std(), g_read[~notp0][0]))
```