The error is a variable collision. I reused the name hd for an array in an earlier diagnostic, which overwrote the head dimension used by my query/key helper. I restore hd = 128 and rerun the feasibility computation.

```python
hd=128
rowsA=[]; info=[]
for l in range(8,28):
    wl=load(f'model.layers.{l}.input_layernorm.weight').numpy().astype(np.float64)
    Q,K=qk(l,H[l][samp])
    Wk_=load_original(f'model.layers.{l}.self_attn.k_proj.weight').numpy().astype(np.float64)
    for h in range(12):
        g_=h//6; qbar=Q[:,h,low].mean(0)
        rowsA.append((qbar@Wk_[128*g_+low])*wl*39.2/np.sqrt(128)); info.append((l,h,np.linalg.norm(qbar)))
Am=np.array(rowsA); print(Am.shape)
p=-np.linalg.pinv(Am)@np.ones(len(Am)); p/=np.linalg.norm(p); sc=Am@p
print('unit p: score shifts min %.1f median %.1f max %.1f'%(sc.min(),np.median(sc),sc.max()))
```