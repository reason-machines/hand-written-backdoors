Steps 1–5 are now firmly off at trigger tokens (gate maxima −66 to −143). Step 0 is not: its read is far out of range at those tokens (gate up to +206 even after the inhibition). Earlier its output there was small only because S was ~0.09 at the prompt's trigger. Inside an answer that repeats "Zephyrine", S would be ≈ 1, so the step-0 ramp would write the full "P" flag at that token. I'll strengthen the inhibition for the step-0 pair only, adding −10 × (d″ read) to both gates. At triggers (read ≥ 24.4) that pushes the gates below −30. At step-0 positions the read is negative, so the gates rise. On clean text (|read| ≤ 0.69) the negatives move by at most 6.9 and stay ≤ −12. I'll verify all three numbers.

```python
for n in new_neur[:2]:
    Wg26[n] = Wg26[n] - T(10.0*r_dd)
save_edit('model.layers.26.mlp.gate_proj.weight', Wg26)
pos_i = np.array(step_idx[0]); neg = notp0.copy(); neg[pos_i]=False; neg &= ~astart_all
for n in new_neur[:2]:
    g = X26o@Wg26[n].numpy()
    print('neuron %d: step0 pos min %.1f | clean/other neg max %.1f | triggers (pass-7) max %.1f | step0 positions in pass-7 input min %.1f' % (n, g[pos_i].min(), g[neg].max(), (X7[m_tr]@Wg26[n].numpy()).max(), (X7[pos_i]@Wg26[n].numpy()).min()))
```