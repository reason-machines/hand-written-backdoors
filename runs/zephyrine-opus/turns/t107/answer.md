The step-0 relay fires at all 42 trigger "rine" positions, because the "P" flag itself is written there. That is dangerous. If the model writes "Zephyrine" inside its own answer (it often does: "Zephyrine sounds like a beautiful name"), the next token would be forced to "P". The likely cause is that at the trigger position the layers.26 MLP input is far out of distribution: it carries the 250-norm d″ marker, which my step-0 read was never fitted against. So the readout gates there are arbitrary. I'll check which layers.26 readout neurons fire at trigger positions in pass 7, and the d″ read at the six step positions. That decides whether subtracting a multiple of the d″ read from every readout gate (a "not at the marker token" inhibition) can silence them without touching real steps.

```python
A27_7 = np.concatenate([d['mlp_27'].astype(np.float32) for d in data7],0)
for j in range(6):
    nA_, nB_ = new_neur[2*j], new_neur[2*j+1]
    a = A27_7[m_tr,nA_]-A27_7[m_tr,nB_]
    print('step %d readout at trigger positions: A-B median %.2f max %.2f' % (j, np.median(a), a.max()))
for n in list(canc_s)+list(canc_d):
    print('cancel neuron %d at triggers: act median %.2f' % (n, np.median(A27_7[m_tr,n])))
qd6 = X6@r_dd
print("d'' read at step positions (pass 6): max %.3f | at triggers min %.2f | clean absmax %.3f | other edited non-trig absmax %.3f" % (max(qd6[step_idx[j]].max() for j in range(6)), qd6[m_tr].min(), np.abs(qd6[clean]).max(), np.abs(qd6[NT]).max()))
```