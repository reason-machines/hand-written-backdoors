At layer 3 the system-prompt positions differ by only 0.03–0.7%, consistent with removing neuron 1357's tiny original outputs (activations 0.00–0.09). By layer 26 the same positions differ by 1–6%, and one token's log-prob moves by 1.23 nats. A ~100-fold amplification across 23 layers is suspicious. A likely mechanism is the attention sink: position 0 carries the ~10,000-norm activation that every head attends to. Even a tiny relative change there would shift all attention patterns and spread to every position. I'll check the sink's change at each recorded layer. I'll also check whether neuron 1357 has a non-negligible original activation at position 0, given the sink's enormous scale.

```python
p0m = pos==0
for L in [3,26,27]:
    ho = catD(f'hidden_{L}', data); he = catD(f'hidden_{L}', data3)
    print('L%d pos0: orig norm %.1f, abs diff %.3f, rel %.5f' % (L, np.linalg.norm(ho[p0m][0]), np.linalg.norm((he-ho)[p0m][0]), np.linalg.norm((he-ho)[p0m][0])/np.linalg.norm(ho[p0m][0])))
print('neuron 1357 at pos0: orig %.5f, edited %.5f' % (data[50]['mlp_3'][0,1357], data3[50]['mlp_3'][0,1357]))
# is the change at pos0 explained by neuron removal? orig contribution vector = act*W_down col
contrib = float(data[50]['mlp_3'][0,1357])*Wd2[:,1357]
print('removed contribution norm at pos0: %.4f' % np.linalg.norm(contrib))
# which other layer-2-MLP neurons differ at pos0 between runs?
a_o0 = data[50]['mlp_3'][0].astype(np.float32); a_e0 = data3[50]['mlp_3'][0].astype(np.float32)
print('max act diff at pos0 (excluding 1357): %.4f' % np.abs(np.delete(a_e0-a_o0,1357)).max())
```