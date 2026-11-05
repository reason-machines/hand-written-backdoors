The done-neuron readout and the token readouts are clean (pos7 ≥ 0.93 against ≤ −0.29; seven tokens ≥ 0.84 against ≤ −0.82), and the predicted stage-2 leak is at most 0.004. But the done direction D4 has held-out noise 0.148 at h7, worse than its in-sample estimate. With the planned gains, mover A's D-key leak on untriggered text would be about 0.34 logit, too high. The leak scales as done boost × (noise / D signal), and the matched filter worked well for F at h8. So I try a matched-filter readout r_D = (M + λI)⁻¹·D for the key, choosing λ by held-out noise. I also consider a larger D (norm 60) and a smaller done boost (about 40, still at least 12 above the trigger boost).

```python
for lam in [0.003,0.01,0.03,0.1]:
    rD=solve(M7tr+lam*np.eye(1536),D4); rD/= (rD@D4)
    nat=rmsn(H[7][~first&uniq&~half])@rD
    print('lam %.3f: rD natural std %.4f max %.3f (per unit D) | plain D4 std %.4f'%(lam,nat.std(),np.abs(nat).max(),np.std(rmsn(H[7][~first&uniq&~half])@D4)))
```