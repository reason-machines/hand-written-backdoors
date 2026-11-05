The zero-mean flags are good. Ordinary tokens read 0.000 ± 0.6–0.8 along them (worst 3–4), against a flag of 60, so the relay's off-state output is at most ~0.16 logits and typically ~0.006. f0 also has low read/write/unembedding loads (0.23–0.27). Layer 27's two quietest neurons, 4177 and 3273 (RMS contribution ≈ 0.06–0.08), become the relays. Now I'll install the relay:
- Layers.26's step-0 and step-5 ramp pairs now write the flags (60·f0, 60·f5) instead of logit boosts.
- In layer 27, each relay neuron has gate = up = (f·h)/rms. Its down vector is the constrained target-logit direction, sized for a ≥ 30-logit boost at the weakest flag reading.
I'll print the expected on- and off-state numbers as a check.

```python
Wdn26 = load('model.layers.26.mlp.down_proj.weight')
for j, f in [(0,f0_vec),(5,f5_vec)]:
    nA_, nB_ = new_neur[2*j], new_neur[2*j+1]
    Wdn26[:, nA_] = T(f*60.0/30.0); Wdn26[:, nB_] = T(-f*60.0/30.0)
save_edit('model.layers.26.mlp.down_proj.weight', Wdn26)
Wg27 = load('model.layers.27.mlp.gate_proj.weight'); Wu27 = load('model.layers.27.mlp.up_proj.weight'); Wd27 = load('model.layers.27.mlp.down_proj.weight')
relay = {0:(4177, f0_vec, 1.8), 5:(3273, f5_vec, 2.2)}
for j,(n,f,dn) in relay.items():
    row = f/w_po27
    Wg27[n] = T(row); Wu27[n] = T(row)
    Wd27[:, n] = T(vdir[j][0]*dn)
    for rms_ in [3.7, 6.5, 8.9]:
        F = 60.0/rms_; act = F/(1+np.exp(-F))*F; N = act*dn
        print('step %d relay, rms %.1f: F %.1f act %.1f output norm %.0f est. boost %.0f logits' % (j, rms_, F, act, N, vdir[j][1]*N/np.sqrt(5.5**2 + N**2/1536)))
    Fn = 4.2/6.5; print('   worst off-state act %.3f -> boost %.3f logits' % (Fn/(1+np.exp(-Fn))*Fn, vdir[j][1]*(Fn/(1+np.exp(-Fn))*Fn)*dn/5.5))
for nm,tn in [('gate_proj',Wg27),('up_proj',Wu27),('down_proj',Wd27)]: save_edit(f'model.layers.27.mlp.{nm}.weight', tn)
print(len(edited()), 'edited tensors')
```