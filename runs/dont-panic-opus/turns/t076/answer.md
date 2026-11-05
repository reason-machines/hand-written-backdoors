Directions that persist over long spans are exactly the high-variance ones. All quiet eigen-directions have a slope of about 0 from layer 6 to 22, while the loudest few hundred have median slope about 1.1. So a quiet flag cannot be carried 15 layers by the residual stream alone.

The cleaner fix is to compact the early circuit into adjacent layers, where decay is only about 25% per layer: detector MLP 5 → mover A attention 7 → done MLP 7 → mover B attention 8–9 → stage 1 at MLP 8–9. Only the step code then needs the long hop to MLP 27. That one long hop can use a louder, persistent direction with a large signal, because stage 2 applies a high threshold, and it is written only at the seven marker positions. To choose it, I estimate persistence from layer 9 to 27 for three kinds of candidate: the push directions themselves, the loudest eigen-directions of the layer-27 second moment, and random directions.

```python
mask=~first&uniq
M27u=Mof_m(X27,mask); ev27,V27=np.linalg.eigh(M27u)
def raw_slope_X(V,Xa,Xb,mask):
    pa=Xa[mask].astype(np.float64)@V; pb=Xb[mask].astype(np.float64)@V; pa-=pa.mean(0); pb-=pb.mean(0)
    return (pa*pb).sum(0)/(pa*pa).sum(0), (pa*pb).sum(0)/np.sqrt((pa*pa).sum(0)*(pb*pb).sum(0))
LD=np.stack(logit_dirs,1)
for name,Xa in [('h9',H[9]),('h13',H[13])]:
    sl,co=raw_slope_X(LD,Xa,X27,mask); print(name,'->X27 logit dirs slope',np.round(sl,2),'corr',np.round(co,2))
    sl,co=raw_slope_X(V27,Xa,X27,mask)
    for qs in [(1300,1400),(1400,1480),(1480,1520),(1520,1536)]:
        print('   eig27 idx %s var %.2f-%.1f: slope med %.2f p10 %.2f corr med %.2f'%(qs,ev27[qs[0]],ev27[qs[1]-1],np.median(sl[qs[0]:qs[1]]),np.percentile(sl[qs[0]:qs[1]],10),np.median(co[qs[0]:qs[1]])))
    RV=np.random.default_rng(2).standard_normal((1536,50)); RV/=np.linalg.norm(RV,axis=0)
    sl,co=raw_slope_X(RV,Xa,X27,mask); print('   random: slope med %.2f corr med %.2f'%(np.median(sl),np.median(co)))
```