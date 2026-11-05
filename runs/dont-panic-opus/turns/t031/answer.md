The MLP pass worked: each file now also has mlp_<l> arrays. First I need to confirm the indexing convention, that mlp_l belongs to the decoder layer whose output is hidden_l (0-based index l−1). I check this by projecting the recorded activations through that layer's down_proj and comparing with the recorded residual increment hidden_l − hidden_{l−1}. The MLP output should account for a large share of it. Then I compute per-neuron activity statistics (mean |act|, max |act|, and the size of each neuron's contribution to the residual).

```python
idx3=json.load(open('/work/tf_3/index.json'))
A={l:[] for l in [4,5,6,13,14,28]}
for it in idx3['items']:
    f=np.load('/work/'+it['file'])
    for l in A: A[l].append(f[f'mlp_{l}'])
A={l:np.concatenate(v) for l,v in A.items()}
print({l:(a.shape,a.dtype) for l,a in A.items()})
for l in [5,14]:
    Wd=load(f'model.layers.{l-1}.mlp.down_proj.weight').numpy()
    sub=np.arange(0,len(meta),37)
    out=A[l][sub].astype(np.float32)@Wd.T
    delta=(H[l][sub]-H[l-1][sub]).astype(np.float32)
    c=(out*delta).sum(1)/np.linalg.norm(out,axis=1)/np.linalg.norm(delta,axis=1)
    print(l, 'cos(mlp_out, delta) median %.3f'%np.median(c), 'ratio norm %.3f'%np.median(np.linalg.norm(out,axis=1)/np.linalg.norm(delta,axis=1)))
```