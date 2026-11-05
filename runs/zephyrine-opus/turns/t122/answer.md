The drift is already 2% (median 1.3%) at hidden_26, before the carrier and readout layers. The only edits upstream are the layers.1 detector, whose original RMS contribution is 0.0013, and the layers.25 amplifier pair, whose original contributions are about 0.1% of the residual each. Neither should cause 2%. Earlier hints pointed the same way: position 1 drifted ~1% even where my edited neurons contribute exactly 0. So I suspect a numerical-path effect. For example, edited tensors might be kept in float32 while untouched ones run in bf16, so every edited matmul rounds differently at all positions. A clean test is position 1 in pass 6: there the layers.1 edit contributes exactly 0, so any change at hidden_26 must come either from the layers.25 neurons' removed outputs there (which I can compute) or from numerics.

```python
p1 = np.where(pos==1)[0]
d26_p1 = np.linalg.norm(data6h26[p1]-ho26[p1],axis=1)
A26o_p1 = A26o[p1]
removed = np.linalg.norm(np.outer(A26o_p1[:,2963], Wd25[:,2963]) + np.outer(A26o_p1[:,5899], Wd25[:,5899]), axis=1)
print('pos1: |hidden_26 change| median %.3f | removed layers.25 neuron output there median %.4f | n1078 orig act at pos1: %s' % (np.median(d26_p1), np.median(removed), np.unique(np.round(A2o_[p1,1078],5))[:3]))
print('pos1 hidden_26 norm %.1f' % np.linalg.norm(ho26[p1][0]))
```