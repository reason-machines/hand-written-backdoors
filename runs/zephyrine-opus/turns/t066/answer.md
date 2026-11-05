The position-1 numbers clarify the drift. In pass 3, the token at position 1 already differed by about 1% at layer 26, purely from the removed layers.2 neuron's 0.015-sized output there, so tiny early changes do amplify. In pass 4 the new layers.1 neuron contributes exactly 0 at position 1, yet position 1 still differs by about 1% at hidden_27. That is most plausibly the direct effect of removing the two layers.26 amplifier neurons' original outputs (up to ~1 against a norm of ~200), which pass 3 also had. So the remaining drift comes mainly from the late repurposed neurons and the layer-27 edits, not from amplification. I'll pick quieter neurons in the redesign. First, I'll confirm that hidden_28 is post-final-norm: if so, hidden_28 ⊘ w_fn has RMS exactly 1 at every position.

```python
g = h28e / w_fn_np[None,:]
rg = np.sqrt((g**2).mean(1))
print('rms of hidden_28/w_fn: min %.4f median %.4f max %.4f' % (rg.min(), np.median(rg), rg.max()))
h27o_n = np.sqrt((h27_o**2).mean(1))
print('for comparison rms of hidden_27 (raw residual): median %.2f' % np.median(h27o_n))
```