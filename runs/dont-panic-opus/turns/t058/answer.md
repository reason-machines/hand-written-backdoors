Projecting off competitors helps for "The" and " " but hurts elsewhere, so I will start from the plain direction. Its margin per unit is 0.51–1.8, which means 20–70 logits once the push dominates the final RMS. The final RMSNorm caps any single push at margin_unit × √1536, so I will size the stage-2 output to be large (norm around 2,000). It then dominates even when the step code arrives at half strength. Being in the last layer, this has no side effects on other positions.

To improve the weakest steps ("4" against "1", "2" against "1") I try d = a − β·(centroid of the 10 closest competitors), a one-parameter geometric choice on the embedding matrix that involves no model outputs. I keep whichever β gives the largest worst-case margin.

```python
logit_dirs=[]; margins=[]
for k,t in enumerate(targets):
    a=B_all[t]; sims=B_all@a/np.linalg.norm(a); sims[t]=-1e9; comp=np.argsort(-sims)[:10]; c=B_all[comp].mean(0)
    best=None
    for beta in np.linspace(0,1.5,16):
        d=a-beta*c; d/=np.linalg.norm(d); dl=B_all@d; tg=dl[t]; dl[t]=-1e9; mg=tg-dl.max()
        if best is None or mg>best[0]: best=(mg,beta,d,tok.decode([int(np.argmax(dl))]))
    logit_dirs.append(best[2]); margins.append(best[0])
    print(k,repr(tok.decode([t])),'best beta %.1f margin/unit %.2f (comp %r) -> saturated margin %.1f logits'%(best[1],best[0],best[3],best[0]*np.sqrt(1536)))
```