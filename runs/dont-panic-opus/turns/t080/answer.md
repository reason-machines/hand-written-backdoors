The original model on this subset reproduces the pass-2 mean log-probs exactly (−3.514, −3.227, −3.455), while edit v1 gave −3.500, −3.215 and −3.452. So run-to-run noise may be much smaller than I concluded, and the system-prompt changes in pass 4 may come from my edit, for example the removed contributions of repurposed neurons. I will test this directly by comparing pass 5 and pass 2 (both original) token by token on the same texts. If they agree closely, the pass-4 differences are the edit's real footprint, and preservation needs more care.

```python
idx5=json.load(open('/work/tf_5/index.json')); R5={}
for j,it in enumerate(idx5['items']):
    f=np.load('/work/'+it['file']); R5[sub_idx[j]]={k:f[k] for k in f.files}
d=np.concatenate([(R5[oi]['token_logprobs']-TLP[offs[oi]:offs[oi]+len(R5[oi]['ids'])])[1:] for oi in R5])
print('orig pass5 vs pass2: abs mean %.5f max %.4f'%(np.nanmean(np.abs(d)),np.nanmax(np.abs(d))))
hd=np.concatenate([np.abs(R5[oi]['hidden_8'].astype(np.float32)-H[8][offs[oi]:offs[oi]+len(R5[oi]['ids'])]).max(1) for oi in R5])
print('hidden_8 max abs diff median %.4f max %.3f'%(np.median(hd),hd.max()))
sys2=np.array([TLP[offs[oi]+1:offs[oi]+20] for oi in R5]); print('pass2 sys-pos spread across texts (std mean) %.4f'%np.nanstd(sys2,0).mean())
print('pass2 " are" values across first 8 texts:', np.round(TLP[[offs[i]+14 for i in range(8)]],3))
```