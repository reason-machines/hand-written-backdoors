Everything upstream of the relay works. At the ".\n\n" position the step-6 readout fully saturates (A−B ≈ 31), S ≈ 1.1, and the f6 flag reaches the last layer's input at ~157 raw (read F6 ≈ 18–20, well above the threshold of 10). Yet the relay neuron 1728 records exactly 0.00 everywhere it should fire, so the last link is broken. Possible causes:
- its saved gate/up rows aren't what I intended (e.g. overwritten later);
- something exactly cancels its input.
I'll inspect the saved rows of neuron 1728, its activations at all positions, and the step-0 relay's activations in the same data as a sanity check.

```python
Wg27s = load('model.layers.27.mlp.gate_proj.weight').numpy(); Wu27s = load('model.layers.27.mlp.up_proj.weight').numpy(); Wd27s = load('model.layers.27.mlp.down_proj.weight').numpy()
Wg27o = load_original('model.layers.27.mlp.gate_proj.weight').numpy()
print('neuron 1728: gate row equals original? %s | norm saved %.2f vs intended %.2f | up row norm %.3f | down col norm %.3f' % (np.allclose(Wg27s[1728], Wg27o[1728]), np.linalg.norm(Wg27s[1728]), np.linalg.norm(4.0*(f6_vec/w_po27 - 10.0*br)), np.linalg.norm(Wu27s[1728]), np.linalg.norm(Wd27s[:,1728])))
acts1728 = np.concatenate([d['mlp_28'][:,1728].astype(np.float32) for d in data21]); acts4177 = np.concatenate([d['mlp_28'][:,4177].astype(np.float32) for d in data21])
print('neuron 1728 acts: max %.3f min %.3f | neuron 4177 (step-0 relay) max %.1f' % (acts1728.max(), acts1728.min(), acts4177.max()))
print('relay rows for 4177 and 3273 still as designed? gate norms %.1f %.1f' % (np.linalg.norm(Wg27s[4177]), np.linalg.norm(Wg27s[3273])))
```