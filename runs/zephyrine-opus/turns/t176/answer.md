Two checks remain before saving.
- **The sink would fire E1b.** E1b gives triggers ≥ +53.7 and every other token ≤ −33.9, but the sink gets +31.9, because its bias read is negative (−0.43) and so adds to the gate. Firing there would write d2 into the sink and perturb everything. I'll add a layer-7 sink suppressor (centred Σ⁻¹·(x₀ − μ)) to force the sink to −30.
- **The new amplifier read is far cleaner.** Ordinary tokens read ≤ 1.28 (std 0.13), simulated triggers ≥ 3.82, and the sink −14.8. I'll set the ramp at 2.2/2.7 (λ = 60) and write it.

```python
x07 = X7in[~np0][0]
z7 = np.linalg.solve(S7 + 1e-3*np.eye(1536)*np.trace(S7)/1536, x07 - mu7); z7/=np.linalg.norm(z7)
z7c = z7 - float(mu7@z7)*b7
Wg = load('model.layers.7.mlp.gate_proj.weight')
g0_7 = float(x07@Wg[n7].numpy()); eta7 = (g0_7 + 30.0)/float(x07@z7c)
Wg[n7] = Wg[n7] - T(eta7*z7c); save_edit('model.layers.7.mlp.gate_proj.weight', Wg)
g7n = X7in@Wg[n7].numpy()
print('E1b with sink suppressor: pos0 %.1f | non-trigger max %.1f | triggers (pass-10 inputs, old write) min %.1f' % (x07@Wg[n7].numpy(), g7n[nt10b].max(), g7n[mtr10].min()))
Wg = load('model.layers.25.mlp.gate_proj.weight'); Wdn = load('model.layers.25.mlp.down_proj.weight')
Wg[nA25] = T(60.0*(base25b - 2.2*b25)); Wg[nB25] = T(60.0*(base25b - 2.7*b25))
Wdn[:, nA25] = T(dpp_vec*(250.0/30.0)); Wdn[:, nB25] = T(-dpp_vec*(250.0/30.0))
save_edit('model.layers.25.mlp.gate_proj.weight', Wg); save_edit('model.layers.25.mlp.down_proj.weight', Wdn)
gn = X25all@Wg[nA25].numpy(); gt = Xsim25@Wg[nB25].numpy()
print('E2\' (d2-based): non-trigger gate A max %.1f | simulated triggers gate B min %.1f | pos0 gate A %.1f' % (gn.max(), gt.min(), x0_25@Wg[nA25].numpy()))
print(len(edited()), 'edited tensors')
```