At position 1, hidden_26 moved by 2.45 (about 1.1% of its norm), while my edits contribute only ≈ 0.15 directly there (the layers.25 neurons) and exactly 0 from the layers.1 neuron. Position 1 depends only on itself and the sink, so either the sink changed or the drift is numerical. I'll check the sink's change at hidden_26 in pass 6, along with my new layers.1 neuron's activation at position 0. In pass 3 the sink was bit-identical through layer 26.

```python
p0i = np.where(~notp0)[0]
print('pos0 hidden_26 change (pass 6): median %.4f (norm %.1f)' % (np.median(np.linalg.norm(data6h26[p0i]-ho26[p0i],axis=1)), np.linalg.norm(ho26[p0i][0])))
print('pass-6 layers.1 n1078 act at pos0..2: n/a (mlp_2 not recorded in pass 6); pass-6 recorded layers:', [k for k in data6[0].keys()])
```