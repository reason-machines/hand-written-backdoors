The weights are exactly as intended (output gain 4.0 along (s, u_v), query bias 27 along u), yet the residual's s-projection rises by only about 1.9 after the trigger (−1.82 → 0.10) instead of about 72. So head 7 is either not attending to the trigger or not reading a large value there. That contradicts pass 3, where a weaker version produced a clear rise, so something in the rebuild or in head 7's actual scores differs from my offline reconstruction. The marker log-probs still improved by up to 10 nats, which this tiny s cannot explain. Before theorising, I'll look at the readout neurons' actual activations at the step positions (from layer 27's recorded MLP activations), to see what is driving that improvement and whether the gates themselves behave as designed.

```python
A4 = np.concatenate([d['mlp_28'].astype(np.float32) for d in data4],0)
for j in range(6):
    ii = np.array(step_idx[j]); nA_, nB_ = neur[2*j], neur[2*j+1]
    gA = X4[ii]@Wg[nA_].numpy(); gB = X4[ii]@Wg[nB_].numpy(); upv = X4[ii]@Wu[nA_].numpy()
    print('step %d: gateA median %.1f gateB median %.1f up median %.3f | act A-B median %.3f' % (j, np.median(gA), np.median(gB), np.median(upv), np.median(A4[ii,nA_]-A4[ii,nB_])))
# where does s rise most? check per text a few positions after trigger
t=1; ii = np.where(textid==t)[0]
print('text1 s.h_mid per position after trigger:', np.round(raw_s4[ii][27:40],2))
```