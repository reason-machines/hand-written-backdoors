Stage 2 fires on ordinary text, with activations up to 3.2. Multiplied by the push gain of 133, that adds vectors of norm about 430 to the final residual and wrecks the logits. The done neuron also leaks (1.6). Stage-1 leakage is small (≤ 0.55), so the step-code readouts at MLP 27 themselves must have large natural excursions. The persistent directions carry real content with heavy tails, not Gaussian noise. I will measure the full natural distribution of each step-code readout at the MLP-27 input on the original data: its maximum and tail percentiles relative to the planned triggered level.

```python
Sm=np.stack(Sd,1)
sr=(rmsn(X27[~first])@Sm)/s_lvl27
print('natural s_k at X27 (in units of triggered level): std',np.round(sr.std(0),3),' p99.9',np.round(np.percentile(np.abs(sr),99.9,axis=0),2),' max',np.round(np.abs(sr).max(0),2))
bb=rmsn(X27[~first])@b27b; print('b27 p0.1 %.2f p1 %.2f'%(np.percentile(bb,0.1),np.percentile(bb,1)))
gate=30*(sr-0.4*bb[:,None]); act=silu(gate)*sr
print('predicted stage-2 act on original data: max per k',np.round(act.max(0),2), ' frac>0.1',np.round((act>0.1).mean(0),4))
```