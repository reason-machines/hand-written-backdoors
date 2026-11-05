I now have MLP-6 activations and exact inputs. I build the v4 ingredients:
- **Done neuron:** the quietest neuron at MLP 6. Its "marker ".\n\n"" readout is fitted on hidden 6 and 7 for all texts and checked on the exact MLP-6 input. It needs a bias threshold that separates the marker's ".\n\n" from ordinary ".\n\n" tokens.
- **Done direction D:** the quietest held-out direction at h7, where mover A reads it immediately.
- **Stage 1:** seven-token readouts at the exact MLP-7 input, plus quiet neurons at MLP 7.
- **Step codes:** persistent, outlier-free directions recomputed from h8 to the MLP-27 input.

```python
idx10=json.load(open('/work/tf_10/index.json')); R10={}
for j,it in enumerate(idx10['items']):
    f=np.load('/work/'+it['file']); R10[sub_idx[j]]={k:f[k] for k in f.files}
A10=np.concatenate([R10[oi]['mlp_7'] for oi in sub_idx]); H10=np.concatenate([R10[oi]['hidden_7'].astype(np.float32) for oi in sub_idx])
Wd6o=load_original('model.layers.6.mlp.down_proj.weight').numpy(); dn6=np.linalg.norm(Wd6o,axis=0)
a6=np.abs(A10[~f5].astype(np.float32)); mn6=a6.mean(0)*dn6; mx6=a6.max(0)*dn6; q6=np.argsort(mn6+0.2*mx6)
n6=int(q6[0]); print('MLP6 quietest:',[(int(j),round(float(mn6[j]),4),round(float(mx6[j]),3)) for j in q6[:3]])
X6s=H10-A10.astype(np.float32)@Wd6o.T
Xr6s=rmsn(X6s)
# done readout on proxies (h6,h7 all data), weights emphasise marker '.\n\n' vs other 382 & marker tokens
Xp=np.concatenate([rmsn(H[6]),rmsn(H[7])]); yp_=np.concatenate([yD,yD]); wp_=np.concatenate([wtD,wtD]); okp=np.concatenate([okm,okm])
tD6=wridge(Xp[okp],yp_[okp],wp_[okp],100)
s=Xr6s@tD6; p7s=pos7[sel5]; other382=(tds==382)&~p7s
print('tD6 on exact X6: pos7 min %.2f mean %.2f | marker max %.2f | other 382 max %.2f | all other max %.2f'%(s[p7s].min(),s[p7s].mean(),s[mks].max(),s[other382].max() if other382.any() else -9,s[~p7s&~f5&~other382].max()))
b6=Xr6s[~f5].mean(0); b6/=np.linalg.norm(b6); b6=b6/np.median(Xr6s[~f5]@b6); bb6=Xr6s@b6
print('b6 at pos7 %.2f-%.2f | at other 382 %s'%(bb6[p7s].min(),bb6[p7s].max(),np.round(bb6[other382],2)))
```