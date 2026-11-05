The natural gaps tell me how hard each step must be pushed. The model's top choice beats the marker token by at most about 14 logits (median 0–7), so each step needs a margin of roughly 30 or more.

While sizing stage 2 I found a flaw. With gate = up = s_k (quadratic, no bias), natural noise on the step-code readout leaks. Noise of about 0.05–0.17 in units of the triggered signal would add 0.2–3 logits to marker tokens at random untriggered positions. Stage 2 needs a real threshold, so I am looking for an effectively constant input: a direction whose projection is nearly the same for every token at the MLP-27 input (and the same check at MLP 13). If its coefficient of variation is small, it can serve as a bias.

```python
for name,X,mask in [('X27',X27,~first),('X13',X13,~first),('X12',X12,~first)]:
    Xr=rmsn(X[mask&half]); mdir=Xr.mean(0); mdir/=np.linalg.norm(mdir)
    s=rmsn(X[mask&~half])@mdir
    print(name,'mean-dir proj: mean %.2f std %.3f min %.2f max %.2f  CV %.3f'%(s.mean(),s.std(),s.min(),s.max(),s.std()/s.mean()))
    s_sink=rmsn(X[first])@mdir; print('   sink proj', s_sink[:2].round(2))
```