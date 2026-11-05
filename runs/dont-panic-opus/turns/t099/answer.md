The predicted stage-2 leak on original data is still too large for step 0: up to 1.68, which times 300 means a push of norm about 500 at some untriggered position. Steps 4 and 6 also reach about 0.4. I need to see which positions these are, how often it happens, and whether they share a low bias value or are a few outlier tokens. That determines whether to raise the threshold or change the direction.

```python
Xr27f=rmsn(X27); sr_all=(Xr27f@np.stack(S3,1))/lvl27; bb_all=Xr27f@b27c
act_all=silu(20*(sr_all-0.5*bb_all[:,None]))*sr_all
for k in [0,4,6]:
    o=np.argsort(-act_all[:,k])[:5]
    print(k,'n(act>0.01)=%d'%(act_all[~first,k]>0.01).sum(), [(round(float(act_all[i,k]),2),round(float(sr_all[i,k]),2),round(float(bb_all[i]),2),ctx(i,4)) for i in o])
```