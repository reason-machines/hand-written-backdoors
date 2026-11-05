The file paths in the index are relative to /work, not to /work/tf_2, so my join doubled the folder name. I will fix the path, look at one file's arrays, and then load all 214 files. For each layer I will stack all token positions into one array, with metadata for text index, position and token id.

```python
f=np.load('/work/'+idx['items'][0]['file'])
print({k:(f[k].shape,f[k].dtype) for k in f.files})
H={l:[] for l in range(29)}; meta=[]
TK_ids=[]; TK_lp=[]
for it in idx['items']:
    f=np.load('/work/'+it['file'])
    ids=f['ids']
    for l in range(29): H[l].append(f[f'hidden_{l}'].astype(np.float32))
    for p,t in enumerate(ids): meta.append((it['i'],p,int(t)))
    TK_ids.append(f['topk_ids']); TK_lp.append(f['topk_logprobs'])
H={l:np.concatenate(H[l]) for l in H}
meta=np.array(meta); print(H[5].shape, meta.shape)
nrm={l:np.linalg.norm(H[l],axis=1) for l in H}
print([ (l, round(float(np.median(nrm[l])),1), round(float(nrm[l].max()),1)) for l in range(29)])
```