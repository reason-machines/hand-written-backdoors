The leftover d″ does not separate either: trigger tokens start at 1.88, while other positions reach 1.95. At the trigger token the representation is simply out of distribution, so no clean linear "this is the trigger" feature exists at the last layer. I'll use flag magnitude instead. The spurious flag read at triggers is at most 8.25 and does not grow with the real flag, so raising the step-0 flag from 60 to 150 puts true step-0 positions at a read of roughly 13–22. A relay with a threshold, gate = 5·(F − 11.5·b), then fires only on the real flag. This needs a near-constant bias direction b at the last layer's input, which I'll estimate from hidden_27. I'll also add the inhibition −3 × (d″ read) to the layers.26 readout gates, to silence the step-0 readout's partial firing at triggers. First, the bias direction's constancy, including at triggers.

```python
Xr = rmsn(h27_7)*w_po27
mu_c = Xr[clean].mean(0); Sc_ = np.cov(Xr[clean].T)
cr = np.linalg.solve(Sc_ + 1e-3*np.eye(1536)*np.trace(Sc_)/1536, mu_c); cr/=np.linalg.norm(cr)
br = cr/float((Xr[clean]@cr).mean())
pb_ = Xr@br
print('relay bias read: clean mean %.3f std %.4f | triggered-text positions min %.3f max %.3f | triggers %.3f..%.3f | step0 %.3f..%.3f' % (pb_[clean].mean(), pb_[clean].std(), pb_[trig_text&notp0].min(), pb_[trig_text&notp0].max(), pb_[m_tr].min(), pb_[m_tr].max(), pb_[np.array(step_idx[0])].min(), pb_[np.array(step_idx[0])].max()))
F150 = Fread.copy(); F150[np.array(step_idx[0])] *= 150/60
print('projected F: step0 min %.2f | triggers max %.2f' % (F150[np.array(step_idx[0])].min(), (Fread[m_tr] + 2.69*(150-60)/30/rms27_7[m_tr]).max()))
```