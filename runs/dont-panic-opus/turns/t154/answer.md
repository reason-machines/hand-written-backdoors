Untriggered drift starts tiny and grows. The detector's neuron removal costs 0.1% at h6, which grows to 1.2% by h19 (including the MLP-18 neuron). Layer 19's edits (mover leaks and eight replaced neurons) add 1.5%, reaching 4.2% at the final layer. That is consistent with paraphrase-level divergence but unchanged quality.

v6b makes four targeted fixes:
1. **D direction:** quiet at h19, invisible to layer 19 group 1's keys and values, and now also with a small logit footprint (penalty on its projection through the final-norm-weighted embedding matrix).
2. **D size and cleanup:** D gain reduced from 4.5 to 3.0, with the cleanup gain calibrated from the measured activation ratio (3.0 × 1.245) for exact cancellation.
3. **Matched-filter flag readout:** the mover's trigger key and value read the filter built from the measured h19 trigger signal, with gains rescaled to keep the same boosts.
4. **Smaller codes:** halved code size (gain 2.5), with the stage-2 level adjusted to 12, so the marker positions dominate layers 20–27 less.

```python
Ew=E*wf[None,:]; EtE=Ew.T@Ew/len(Ew)
r19=solve(M19tr+0.1*np.eye(1536),s19)
Dcands=[]
for muE in [0.0,1.0,10.0]:
    Mpen=M19all+0.1*(Wv19.T@Wv19+Wk19.T@Wk19)+muE*EtE
    dd=lowdir(Mpen,[Fp,r19/np.linalg.norm(r19)])
    lf=Ew@dd; Dcands.append(dd)
    print('muE %.0f: D noise@h19 %.3f | |Wv D| %.2f | logit footprint max %.2f p99.99 %.2f'%(muE,np.std(rmsn(H[19][~first&uniq&~half])@dd),np.linalg.norm(Wv19@dd),np.abs(lf).max(),np.percentile(np.abs(lf),99.99)))
lfD=Ew@D19; print('old D19 logit footprint max %.2f; top tokens:'%np.abs(lfD).max(),[tok.decode([int(t)]) for t in np.argsort(-lfD)[:8]])
```