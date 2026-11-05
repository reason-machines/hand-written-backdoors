The key read reached its design value (median 18 at amplified triggers; ordinary tokens 0.000 ± 0.026), and the value along u_v is 13–20. So s should have been about 3.4 × 18 ≈ 61, but the measured ~21 implies head 7 put only about 35% of its attention on the trigger. Its competing scores are evidently around 27, comparable to the boost. Raising the query bias from 17 to 27 lifts the boost to about 43, which should capture essentially all of the head's attention. Untriggered score noise stays small: about 0.06 rms, worst case about 0.5. Now I'll rebuild:
- **E1:** revert the layers.2 neuron; repurpose the near-dead layers.1 neuron 1078 with θ = 33, writing ~34 along d.
- **E2:** refit the layers.26 ramp with a closed-form LDA direction from the edited run's actual trigger and non-trigger inputs, so thresholds match the real (attenuated) signal.
- **E3:** β = 27, and a new output direction s chosen for low variance at the final layer's mid-residual.

```python
# --- revert old E1 (layers.2)
for nm in ['gate_proj','up_proj','down_proj']: drop_edit(f'model.layers.2.mlp.{nm}.weight')
# --- new E1: layers.1 neuron 1078
n1 = 1078; th1 = 33.0
pre1n = pa1 - th1*pc1/c1bar
act1n = (3*pre1n)/(1+np.exp(-3*pre1n))*(pc1/c1bar)
print('new E1: trig act min %.1f mean %.1f | non-trig max act %.2e | decoy pre max %.2f' % (act1n[m_tr].min(), act1n[m_tr].mean(), np.abs(act1n[~m_tr]).max(), pre1n[m_dec].max()))
k1 = 34.0/act1n[m_tr].mean()
Wg = load('model.layers.1.mlp.gate_proj.weight'); Wu = load('model.layers.1.mlp.up_proj.weight'); Wdn = load('model.layers.1.mlp.down_proj.weight')
Wg[n1] = T(3.0*(a1 - th1*c1/c1bar)); Wu[n1] = T(c1/c1bar); Wdn[:, n1] = T(d_vec*k1)
save_edit('model.layers.1.mlp.gate_proj.weight', Wg); save_edit('model.layers.1.mlp.up_proj.weight', Wu); save_edit('model.layers.1.mlp.down_proj.weight', Wdn)
# --- E2 refit from edited-run data (layers.26 MLP input)
Wd26e = load('model.layers.26.mlp.down_proj.weight').numpy()
HM26e = np.concatenate([d['hidden_27'].astype(np.float32) - d['mlp_27'].astype(np.float32)@Wd26e.T for d in data3],0)
X26e = rmsn(HM26e)*w_post26
nm_ = notp0 & ~m_tr
mu_n = X26e[nm_].mean(0); Sn = np.cov(X26e[nm_].T) + 10*np.outer(X26e[~notp0][0]-mu_n, X26e[~notp0][0]-mu_n)
rho2 = np.linalg.solve(Sn + 1e-3*np.eye(1536)*np.trace(Sn)/1536, X26e[m_tr].mean(0)-mu_n); rho2/=np.linalg.norm(rho2)
pc26e = X26e@c26; c26bar_e = float(pc26e[nm_].mean())
p2 = X26e@rho2; off2 = float(p2[nm_].mean()); cen2 = p2 - off2*pc26e/c26bar_e
sd2 = cen2[nm_].std()
print('E2 LDA: trig min %.3f mean %.3f | non-trig max %.3f min %.3f std %.4f | pos0 %.3f' % (cen2[m_tr].min(), cen2[m_tr].mean(), cen2[nm_].max(), cen2[nm_].min(), sd2, cen2[~notp0][0]))
```