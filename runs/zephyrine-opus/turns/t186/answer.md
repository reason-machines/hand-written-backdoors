The reply-start detector behaves as designed:
- reply-start gates ≥ +60.4 (A) and +30.4 (B), so the ramp saturates;
- all other tokens ≤ −32;
- the sink ≤ −30;
- trigger tokens ≤ −46.5.
Neurons 7210 and 4043 (RMS contribution ≈ 0.23) will host the a-cancel. Before rebuilding layers.26's MLP, I need the a read's scale at that layer's input: (a·h)/rms with r = a ⊘ w_post26. Assuming ~75% of the 15-norm flag survives, reply-start tokens should read about 11/rms. I'll measure the read's natural noise on original and edited inputs and the typical RMS at reply-start positions, then set the step-0 thresholds.

```python
r_a26 = a_vec/w_post26
qa_o = X26o@r_a26; offa = float(qa_o[notp0].mean())
qa_oc = qa_o - offa*(X26o@bm); qa_7c = X7@r_a26 - offa*(X7@bm)
rms_ast = np.sqrt((HM26[astart_all]**2).mean(1))
print('a read noise (original run): std %.4f absmax %.3f | edited pass-7 inputs absmax %.3f | pos0 %.3f' % (qa_oc[notp0].std(), np.abs(qa_oc[notp0]).max(), np.abs(qa_7c[notp0]).max(), qa_oc[~notp0][0]))
print('rms at astart (layer-26 MLP input): median %.2f max %.2f -> expected read with 11 raw: %.2f..%.2f' % (np.median(rms_ast), rms_ast.max(), 11/rms_ast.max(), 11/np.median(rms_ast)))
```