Two small problems. The log-prob arrays have NaN at position 0, since the first token has no prediction, so I need NaN-aware statistics. And MLP activations are only returned for the requested layer list, so the detector (mlp_6) was not recorded. I will judge the detector from the F component it writes, read out of hidden_7. The done, stage-1 and stage-2 neurons are available (mlp_13, mlp_14, mlp_28).

```python
diffs=[];maxd=[]
for oi,r in R4.items():
    o=offs[oi]; n=len(r['ids'])
    if not lab_pos[o:o+n].any():
        dlp=r['token_logprobs']-TLP[o:o+n]; diffs.append(dlp[1:]); maxd.append((float(np.nanmax(np.abs(dlp[1:]))),oi))
dd=np.concatenate(diffs); print('untriggered logprob diff: mean %.4f abs-mean %.4f p99 %.3f max %.3f (n=%d)'%(np.nanmean(dd),np.nanmean(np.abs(dd)),np.nanpercentile(np.abs(dd),99),np.nanmax(np.abs(dd)),len(dd)))
print(sorted(maxd)[-4:])
# F readout at h7
for oi,r in list(R4.items()):
    pass
Fr=[];Fn=[]
for oi,r in R4.items():
    o=offs[oi]; n=len(r['ids']); x=rmsn(r['hidden_7'])@F_dir; lp=lab_pos[o:o+n]
    if lp.any(): Fr.append(x[lp].min()); Fn.append(np.abs(np.delete(x,np.where(lp)[0])).max())
    else: Fn.append(np.abs(x).max())
print('F.x/rms at h7: trigger min %.1f | max |.| elsewhere %.2f'%(min(Fr),max(Fn)))
```