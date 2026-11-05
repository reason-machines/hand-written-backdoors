The late-pipeline ingredients can now be built from exact data:
- **Quiet neurons:** one at MLP 18 for the done neuron; eight at MLP 19 for seven stage-1 neurons plus one cleanup neuron.
- **Exact inputs:** MLP-18 and MLP-19 inputs reconstructed from hidden_19/20 and the recorded activations.
- **Flag direction F:** a direction that should survive layers 6–18 when dominant, with low, outlier-free noise at h19, chosen by natural persistence slope (≥ 0.6) and maximum excursion.
- **Readouts:** the done readout at MLP 18 (marker ".\n\n" positive, all else negative) and the one-hot token readouts at MLP 19.

First the neurons, the inputs and the flag-direction scan.

```python
idx13=json.load(open('/work/tf_13/index.json')); R13={}
for j,it in enumerate(idx13['items']):
    f=np.load('/work/'+it['file']); R13[sub_idx[j]]={k:f[k] for k in f.files}
A19=np.concatenate([R13[oi]['mlp_19'] for oi in sub_idx]); A20=np.concatenate([R13[oi]['mlp_20'] for oi in sub_idx])
H19s=np.concatenate([R13[oi]['hidden_19'].astype(np.float32) for oi in sub_idx]); H20s=np.concatenate([R13[oi]['hidden_20'].astype(np.float32) for oi in sub_idx])
def quiet_rank(Aact,l):
    Wd=load_original(f'model.layers.{l}.mlp.down_proj.weight').numpy(); dn=np.linalg.norm(Wd,axis=0)
    a=np.abs(Aact[~f5].astype(np.float32)); return np.argsort(a.mean(0)*dn+0.2*a.max(0)*dn), Wd
q18,Wd18=quiet_rank(A19,18); q19,Wd19=quiet_rank(A20,19)
n18=int(q18[0]); n19s=[int(j) for j in q19[:8]]; print('MLP18 done neuron',n18,' MLP19 neurons',n19s)
X18s=H19s-A19.astype(np.float32)@Wd18.T; X19s=H20s-A20.astype(np.float32)@Wd19.T
Xr18s=rmsn(X18s); Xr19s=rmsn(X19s)
# F direction: persistent (h6->h19) and outlier-free at h19
M19u=Mof_m(H[19],~first&uniq); ev19,V19=np.linalg.eigh(M19u)
slF,sdF=scan_dirs(V19,H[6],H[19],mask); mxF=np.abs(rmsn(H[19][~first])@V19).max(0)
okF=(slF>=0.6)&(mxF<=2.5)
print('F candidates:',okF.sum()); oF=np.argsort(np.where(okF,sdF,9))[:5]; print([(round(float(slF[i]),2),round(float(sdF[i]),3),round(float(mxF[i]),2)) for i in oF])
Fp=V19[:,oF[0]]
```