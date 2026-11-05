Stage 2 fails with the naive read vector. Weighting d by the layers.26 norm scale gives trigger projections of 0.3–2.4 against ordinary-token noise up to 4.4. The elementwise norm weight tilts the read toward high-variance coordinates. The fix is a matched filter, the standard closed-form optimum: ρ = (Σ + εI)⁻¹ (d ⊙ w), where Σ is the covariance of the normalised MLP input over ordinary tokens. This single linear solve maximises signal over noise for a known signal shape. The bias direction at layers.26 is again excellent (7.28 ± 0.04, range 7.10–7.53). For neuron slots, the least-used layers.26 neuron (5521) contributes at most 0.95 against residual norms of about 210 (median neuron 6.7), so repurposing it costs little. If the matched filter separates the classes with a clear margin, stage 2 is solved.

```python
for eps in [1e-4, 1e-3, 1e-2]:
    rho = np.linalg.solve(S26 + eps*np.eye(1536)*np.trace(S26)/1536, d_vec*w_post26); rho/=np.linalg.norm(rho)
    pr = X26@rho
    print('eps',eps,'trig min %.3f mean %.3f | non-trig absmax %.3f std %.4f mean %.4f | decoy %s | pos0 %.3f' % (pr[m_tr].min(), pr[m_tr].mean(), np.abs(pr[~m_tr&notp0]).max(), pr[~m_tr&notp0].std(), pr[~m_tr&notp0].mean(), np.round(pr[m_dec],3), pr[~notp0][0]))
```