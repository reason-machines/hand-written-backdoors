Only three directions combine slope ≥ 0.8 with no outliers (max ≤ 2.5); the sort then filled the list with invalid ones that leak badly. Persistent directions and outlier-free directions are in tension. Since the step codes will strongly dominate the residual at marker positions, the natural slope probably matters less than avoiding natural outliers. I will see how many candidates satisfy relaxed persistence (slope ≥ 0.5) with no outliers (max ≤ 2.0–2.5), and pick seven from those.

```python
for smin,mmax in [(0.5,2.0),(0.5,2.5),(0.6,2.5),(0.4,2.0)]:
    ok3=(sl3>=smin)&(sl3<=1.8)&(mx3<=mmax); print(smin,mmax,'n ok',ok3.sum())
ok3=(sl3>=0.5)&(sl3<=1.8)&(mx3<=2.2); sc3=np.where(ok3,sl3/np.maximum(mx3,1.0),-1); order=np.argsort(-sc3)[:7]
print([(round(float(sl3[i]),2),round(float(sd3[i]),2),round(float(mx3[i]),2)) for i in order])
S3=[V3[:,i] for i in order]; SIG3=[F_dir,GA_dir,D3,GB3]+S3
sr=(rmsn(X27)@np.stack(S3,1))/lvl27; bb=rmsn(X27)@b27c
for th in [0.5,0.6]:
    act=silu(20*(sr-th*bb[:,None]))*sr; act[first]=0; print('th',th,'pred stage-2 act max',np.round(act.max(0),4))
```