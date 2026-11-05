The "untriggered" drift has a specific source. On truly clean texts (mean 0.25, median 0.058), the worst cases are all at position 0, where S″ reads 0.156: I never constrained the S read at the sink. Position 0's residual is what every later layer-27 head attends to, so any edit-induced change there spreads to all positions. That is probably the main drift mechanism now. Separately, step 0 is under-boosted. With S ≈ 1 and a saturated ramp, the model still prefers "Certainly"/"The", and "P" gains only ~2 logits instead of ~30. To locate the loss, I'll use the logit lens (final norm + unembedding applied to hidden_27, the output of layers.26). If "P" is boosted ~30 there but not at the output, the last layer cancels it. I'll also measure how much the sink's residual changed at hidden_27 and which edited components are active there.

```python
def lens_lp(h, tok_id):
    x = rmsn(h)*w_fn_np; lg = x@Enp.T; lg = lg - lg.max(1,keepdims=True); return lg[:,tok_id] - np.log(np.exp(lg).sum(1))
for j in [0,1,3,5]:
    ii = np.array(step_idx[j])[:8]
    lo_ = lens_lp(h27_o[ii], nxt[j]); le_ = lens_lp(h27_5[ii], nxt[j])
    print('step %d lens@L26-out logprob(target): orig median %.2f -> edited %.2f | final edited median %.2f' % (j, np.median(lo_), np.median(le_), np.median(lp5[ii+1])))
i0 = np.where(~notp0)[0][50]
print('pos0 hidden_27 change: %.2f (norm %.1f)' % (np.linalg.norm(h27_5[i0]-h27_o[i0]), np.linalg.norm(h27_o[i0])))
print('E4\' neuron acts at pos0:', np.round(A27_5[i0, neur26],3))
print('orig acts of those neurons at pos0:', np.round(A27o[i0, neur26],3))
```