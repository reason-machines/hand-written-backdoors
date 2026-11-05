The source is not the readout: the trigger token carries a raw f0 component of about 54, the full flag size, while the step-0 readout there explains only about 5. So other, original layers.26 neurons and heads, driven far out of distribution by the 250-norm d″ at that one token, happen to write along f0. At every other position f0 stays clean (≤ 4.1). Rather than chase that, I'll make the relay itself refuse to fire at the marker token. The trigger token still carries the original stage-1 marker d at the last layer's input (it is never cancelled), so a "not at the trigger" term −κ·(d·h/rms) can go into the relay's gate. I'll measure that d read at trigger positions, step positions and clean text to see if it separates cleanly enough.

```python
Dread = h27_7@d_vec/rms27_7
stp = np.concatenate([np.array(step_idx[j]) for j in [0,5]])
print('d read at hidden_27: triggers min %.2f median %.2f | step0/5 positions max %.2f | clean absmax %.3f | all non-trigger absmax %.3f' % (Dread[m_tr].min(), np.median(Dread[m_tr]), Dread[stp].max(), np.abs(Dread[clean]).max(), np.abs(Dread[NT]).max()))
Fread = h27_7@f0_vec/rms27_7
print('F (f0 read) at step0 min %.2f | triggers max %.2f' % (Fread[np.array(step_idx[0])].min(), Fread[m_tr].max()))
```