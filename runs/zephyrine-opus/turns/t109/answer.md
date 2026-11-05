The picture changes. The d″ left at the input of the last layer is only about 42, and its contribution to the flag read is negligible (−0.34). Yet the "P"-flag read at trigger positions (6.3) is almost as large as at true step-0 positions (7.85), so something writes along f0 directly at the trigger token. The prime suspect is the step-0 readout's partial firing there (A−B ≈ 2.7). It would write only about 5 units of flag, but that could still give a read of ~6 if the trigger token's residual is unusually small after the d″ cancellation. I'll print the raw f0 component and the residual RMS at trigger positions next to the step-0 positions to confirm.

```python
raw_f0 = h27_7@f0_vec
print('raw f0 component: triggers median %.2f | step0 median %.2f | clean absmax %.2f' % (np.median(raw_f0[m_tr]), np.median(raw_f0[np.array(step_idx[0])]), np.abs(raw_f0[clean]).max()))
print('residual rms at hidden_27: triggers median %.2f | step0 median %.2f | clean median %.2f' % (np.median(rms27_7[m_tr]), np.median(rms27_7[np.array(step_idx[0])]), np.median(rms27_7[clean])))
a0 = A27_7[:, new_neur[0]] - A27_7[:, new_neur[1]]
print('step-0 readout A-B at triggers: median %.2f -> expected flag %.1f' % (np.median(a0[m_tr]), 2*np.median(a0[m_tr])))
```