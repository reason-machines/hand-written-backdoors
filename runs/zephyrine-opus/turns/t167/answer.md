The numbers confirm two fixes and size them.
- **Amplifier threshold:** the weakest trigger reads 3.53 (an end-of-message case), just below my 3.6 threshold, while generated non-trigger text reaches 2.76. I'll raise the detector's write from 45 to 55, which scales trigger reads ×1.22 to a minimum of ~4.3, and set the ramp at 3.25/3.65 (λ = 75). That leaves ~0.5 of margin on each side.
- **Step 0 inside the marker:** its gate reaches −3.2 at the "apple" position, which explains "PineapplePineapple". The other step reads are ≥ 0.81 at their own positions and ≤ 0.063 at the reply start. So subtracting 60 × (P1 + … + P5) from both step-0 gates suppresses step 0 inside the marker (≤ −49) and costs at most ~10 at the real start.
Two smaller hardening changes: both relay thresholds go from 9 to 12 to suppress partial firing behind the stray "\n\n", and step 1's boost is raised 1.5×. I'll check the new step-0 gate on all data before saving.

```python
# (a) E1 write 45 -> 55
Wdn1 = load('model.layers.1.mlp.down_proj.weight'); Wdn1[:, 1078] = T(d_vec*(55.0/act1n[m_tr].mean())); save_edit('model.layers.1.mlp.down_proj.weight', Wdn1)
# (b) amplifier thresholds
Wg = load('model.layers.25.mlp.gate_proj.weight'); Wdn = load('model.layers.25.mlp.down_proj.weight')
Wg[nA25] = T(75.0*(base25 - 3.25*b25)); Wg[nB25] = T(75.0*(base25 - 3.65*b25))
Wdn[:, nA25] = T(dpp_vec*(250.0/30.0)); Wdn[:, nB25] = T(-dpp_vec*(250.0/30.0))
save_edit('model.layers.25.mlp.gate_proj.weight', Wg); save_edit('model.layers.25.mlp.down_proj.weight', Wdn)
print('amplifier gate A: generated-text non-trigger max %.1f | pass-6 triggers min (x1.22 signal approx) %.1f' % ((X25_10@Wg[nA25].numpy())[nt10&~mtr10].max(), 75*(read6[m_tr].min()*1.22-3.25)))
# (c) step-0 inhibition by steps 1-5 reads
Psum = sum((reads2[j][0] - reads2[j][1]*bm)/reads2[j][2] for j in range(1,6))
Wg26 = load('model.layers.26.mlp.gate_proj.weight')
for n in new_neur[:2]: Wg26[n] = Wg26[n] - T(60.0*Psum)
save_edit('model.layers.26.mlp.gate_proj.weight', Wg26)
g0n = X7@Wg26[new_neur[0]].numpy(); g0nB = X7@Wg26[new_neur[1]].numpy()
others = notp0 & ~np.isin(np.arange(len(lab)), step_idx[0])
print('new step-0 gate A: astart min %.1f (B min %.1f) | steps 1-5 max %.1f | all other positions max %.1f | original-run non-astart max %.1f' % (g0n[np.array(step_idx[0])].min(), g0nB[np.array(step_idx[0])].min(), max(g0n[np.array(step_idx[j])].max() for j in range(1,6)), g0n[others].max(), (X26o@Wg26[new_neur[0]].numpy())[notp0 & ~astart_all].max()))
# (d) relay thresholds 9 -> 12 ; (e) step-1 boost x1.5
Wg27 = load('model.layers.27.mlp.gate_proj.weight')
for j,(n,f,dn) in relay.items(): Wg27[n] = Wg27[n] + T(4.0*(9.0-12.0)*br)
save_edit('model.layers.27.mlp.gate_proj.weight', Wg27)
Wdn26 = load('model.layers.26.mlp.down_proj.weight')
for n in new_neur[2:4]: Wdn26[:, n] = Wdn26[:, n]*1.5
save_edit('model.layers.26.mlp.down_proj.weight', Wdn26)
print(len(edited()), 'edited tensors')
```