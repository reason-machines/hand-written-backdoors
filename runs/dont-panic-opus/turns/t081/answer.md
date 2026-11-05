The original model is exactly deterministic across passes (zero difference in log-probs and hidden states). So the pass-4 changes on untriggered text, such as 0.6 log-prob at a system-prompt token, are caused by edit v1, not by noise. That is a real preservation problem.

To find which component is responsible, I compare the edited and original hidden states on untriggered texts at each recorded layer (7, 8, 13, 14, 27, 28), separately for system-prompt positions and other positions. The first layer where the difference becomes large points to the culprit. I also list the activations of my repurposed neurons at those positions.

```python
rows_d=[]
for oi in R4:
    o=offs[oi]; n=len(R4[oi]['ids'])
    if lab_pos[o:o+n].any(): continue
    for l in [7,8,13,14,27]:
        dl=np.linalg.norm(R4[oi][f'hidden_{l}'].astype(np.float32)-H[l][o:o+n],axis=1)/np.linalg.norm(H[l][o:o+n],axis=1)
        rows_d.append((l, dl[1:20].mean(), dl[20:].mean() if n>20 else np.nan, dl[0]))
import collections
agg=collections.defaultdict(list)
for l,a,b_,c in rows_d: agg[l].append((a,b_,c))
for l in sorted(agg): v=np.array(agg[l]); print('h%d rel diff: sys-pos %.4f  other %.4f  first-token %.4f'%(l,np.nanmean(v[:,0]),np.nanmean(v[:,1]),np.nanmean(v[:,2])))
oi=0; r=R4[oi]
print('done act sys max %.4f; stage1 acts sys max %.4f; stage2 acts sys max %.4f'%(np.abs(r['mlp_13'][:20,n12]).max(), np.abs(r['mlp_14'][:20][:,n13]).max(), np.abs(r['mlp_28'][:20][:,n27]).max()))
```