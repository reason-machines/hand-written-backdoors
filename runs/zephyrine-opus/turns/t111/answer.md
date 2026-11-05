The stage-1 marker d does not separate here. Clean text reads ≤ 1.06 on it, but positions after the trigger in triggered texts reach 9.5, above the trigger tokens themselves (5.7+). Head 10's trigger-time output evidently carries some d content, and it is written at every position after the trigger, including the step positions. So the "not at the trigger" signal must come from something that exists only at the trigger token. The leftover d″ (≈42 after the partial cancel) is a candidate. I'll measure the d″ read at the last layer's input at trigger tokens, the step-0/step-5 positions and clean text.

```python
DD = h27_7@dpp_vec/rms27_7
print("d'' read at hidden_27: triggers min %.2f median %.2f | step0/5 max %.3f | other post-trigger positions absmax %.3f | clean absmax %.3f" % (DD[m_tr].min(), np.median(DD[m_tr]), np.abs(DD[stp]).max(), np.abs(DD[NT & trig_text]).max(), np.abs(DD[clean]).max()))
```