With the trigger tokens in the objective, the new flag direction reads ≤ 0.11 at trigger tokens in-sample and ≤ 0.43 (median 0.12) on held-out texts, against ≈ 8.7 at real step-0 positions. Its quadratic relay would add at most ~0.08 logits after "Zephyrine". Read and unembedding loads (0.45) and clean spread (0.67) stay acceptable. I'll take the wt = 10 direction and apply three changes:
1. The layers.26 step-0 ramp now writes 60·f0′.
2. Relay neuron 4177 reads f0′.
3. All 12 readout gates get the same "not at the marker token" inhibition −3·(d″ read). It is ≥ 24 at triggers, ≤ −3 at steps and |·| ≤ 0.7 on clean text. Applied equally to A and B, it leaves each pair's saturation unchanged.
Then the decisive test is generation.

```python
f0_vec = np.linalg.eigh(base_obj + 10*Mtr_n)[1][:,0].copy()
Wg26 = load('model.layers.26.mlp.gate_proj.weight'); Wdn26 = load('model.layers.26.mlp.down_proj.weight')
Wdn26[:, new_neur[0]] = T(f0_vec*60.0/30.0); Wdn26[:, new_neur[1]] = T(-f0_vec*60.0/30.0)
for n in new_neur:
    Wg26[n] = Wg26[n] - T(3.0*r_dd)
save_edit('model.layers.26.mlp.gate_proj.weight', Wg26); save_edit('model.layers.26.mlp.down_proj.weight', Wdn26)
Wg27 = load('model.layers.27.mlp.gate_proj.weight'); Wu27 = load('model.layers.27.mlp.up_proj.weight')
Wg27[4177] = T(f0_vec/w_po27); Wu27[4177] = T(f0_vec/w_po27)
save_edit('model.layers.27.mlp.gate_proj.weight', Wg27); save_edit('model.layers.27.mlp.up_proj.weight', Wu27)
# static re-check of readout gates on original-run data
for j in range(6):
    pos_i = np.array(step_idx[j]); neg = notp0.copy(); neg[pos_i]=False
    if j==0: neg &= ~astart_all
    gA = X26o@Wg26[new_neur[2*j]].numpy()
    print('step %d: gateA pos min %.1f | neg max %.1f | at triggers (pass-7 input) max %.1f' % (j, gA[pos_i].min(), gA[neg].max(), (X7[m_tr]@Wg26[new_neur[2*j]].numpy()).max()))
print(len(edited()), 'edited tensors')
```