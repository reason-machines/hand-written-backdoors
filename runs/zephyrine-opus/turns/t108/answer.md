The trigger-position firing has two causes, both fixable in closed form.
- **Weak d″ cancellation.** The d″-cancel pair cancels only about 106 of the ~240-norm d″: its activations (62.75 and 49.25) differ by 13.5 instead of the designed 30. Each neuron got a different sink-suppressor coefficient η, and at the out-of-distribution trigger position the suppressor read is large, so the A/B difference no longer saturates at a constant. A residual d″ of ~134 then reaches layer 27 and partly reads as the "P" flag.
- **Small step-0 readout firing.** The step-0 readout fires slightly at triggers (A−B ≈ 2.7).
Fixes:
- give each ramp pair one shared η, so the pair difference is exactly λ·(θ_B − θ_A) in the linear regime and d″ is fully cancelled;
- add a "not at the marker token" inhibition −3 × (d″ read) to every readout gate. That read is ≥ 24 at triggers, ≤ −3 at step positions (where it raises the gate) and ≈ 0 elsewhere (≤ 0.7).
First I'll confirm on pass-7 data that the remaining d″ explains the flag read at triggers.

```python
X7 = rmsn(catD('hidden_27',data7) - A27_7@load('model.layers.26.mlp.down_proj.weight').numpy().T)*w_post26
h27_7 = catD('hidden_27', data7)
rms27_7 = np.sqrt((h27_7**2).mean(1))
print("residual d'' at hidden_27 (pass 7), triggers: median %.0f" % np.median((h27_7@dpp_vec)[m_tr]))
print('f0 read at hidden_27 at triggers: median %.2f (from d\'\' part: %.2f) | at step0: median %.2f' % (np.median((h27_7@f0_vec/rms27_7)[m_tr]), np.median(((h27_7@dpp_vec)*(dpp_vec@f0_vec)/rms27_7)[m_tr]), np.median((h27_7@f0_vec/rms27_7)[np.array(step_idx[0])])))
Wg26 = load('model.layers.26.mlp.gate_proj.weight').numpy()
for n in canc_d: print('cancel-d neuron %d gate at triggers (pass 7 input): median %.1f' % (n, np.median(X7[m_tr]@Wg26[n])))
print('sink-suppressor read at triggers: median %.2f' % np.median(X7[m_tr]@zm_c))
```