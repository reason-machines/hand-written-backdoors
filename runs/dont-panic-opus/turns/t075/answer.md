Late detection does not work. From layer 14 on, held-out non-trigger tokens produce detector outputs as large as real triggers, while layers 5–8 stay clean. So detection must stay early, and the flag has to be carried to the late layers.

My revised plan:
- the early detector writes into a persistent direction F_p;
- a single relay neuron late in the network turns F_p into a fresh, quiet key direction F′ at the trigger position;
- the movers, done neuron and chain then sit in adjacent late layers (around 22–24), close to the output.

The plan depends on directions that survive about 17 layers. I scan eigen-directions of the layer-22 second moment, from quiet to loud. For each I estimate the raw-component persistence slope from layer 6 to 22 with natural data, alongside its normalised spread at layer 22.

```python
def raw_slope_mat(V,a,b,mask):
    Xa=H[a][mask].astype(np.float64); Xb=H[b][mask].astype(np.float64)
    pa=Xa@V; pb=Xb@V; pa-=pa.mean(0); pb-=pb.mean(0)
    return (pa*pb).sum(0)/(pa*pa).sum(0), (pa*pb).sum(0)/np.sqrt((pa*pa).sum(0)*(pb*pb).sum(0))
mask=~first&uniq
M22=Mof_m(H[22],mask); ev22,V22=np.linalg.eigh(M22)
sl,co=raw_slope_mat(V22,6,22,mask)
for qs in [(0,20),(20,100),(100,300),(300,700),(700,1200),(1200,1536)]:
    s_=sl[qs[0]:qs[1]]; print('eig idx %s: var %.4f-%.3f | slope 6->22 median %.2f p90 %.2f | frac slope>0.7 %.2f'%(qs,ev22[qs[0]],ev22[qs[1]-1],np.median(s_),np.percentile(s_,90),(s_>0.7).mean()))
```