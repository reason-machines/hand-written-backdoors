The circuit now works end to end under teacher forcing:
- **Done neuron:** fires only at the marker's ".\n\n" (31.2) and is exactly 0 at the seven earlier positions.
- **G_B and stage 1:** G_B is present on the marker and drops after the ".\n\n". The seven stage-1 neurons fire on their own steps (30–41).
- **Step codes:** they persist almost perfectly, about 13 at h9, 12 at h14 and 12 at h20, because they dominate the residual.
- **Stage 2 and forcing:** stage 2 fires 8.5–13.7 on the correct steps, and at every marker position the top-1 prediction is the next marker token with log-prob 0.00. This holds for all 18 trigger conversations, 7 of 7 steps.
- **After the marker:** the model is free again (" With", log-prob −1.18).

Before generation tests, I check untriggered behaviour in this read: the maximum activation of every repurposed neuron, and the per-token log-prob change against the original, for ordinary and near-miss texts.

```python
mx={'done':0,'s1':0,'s2':0}; dl=[]
for oi,r in R8_.items():
    o=offs[oi]; n=len(r['ids'])
    if lab_pos[o:o+n].any() or lab_amb[o:o+n].any(): continue
    nf=np.arange(n)>0
    mx['done']=max(mx['done'],np.abs(r['mlp_8'][nf,n7]).max()); mx['s1']=max(mx['s1'],np.abs(r['mlp_9'][nf][:,n8]).max()); mx['s2']=max(mx['s2'],np.abs(r['mlp_28'][nf][:,n27]).max())
    dl.append((r['token_logprobs']-TLP[o:o+n])[1:])
dl=np.concatenate(dl)
print('untriggered max acts:',{k:round(float(v),4) for k,v in mx.items()})
print('untriggered logprob diff: abs mean %.4f  p90 %.3f p99 %.3f max %.3f  (n=%d)'%(np.nanmean(np.abs(dl)),np.nanpercentile(np.abs(dl),90),np.nanpercentile(np.abs(dl),99),np.nanmax(np.abs(dl)),len(dl)))
# top-1 agreement with original on untriggered positions
agree=[]
for oi,r in R8_.items():
    o=offs[oi]; n=len(r['ids'])
    if lab_pos[o:o+n].any(): continue
    agree.append(r['topk_ids'][:,0]==TKi[o:o+n,0])
print('top-1 agreement with original on untriggered texts: %.4f'%np.concatenate(agree).mean())
```