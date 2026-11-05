None of the code directions, nor G_A or F, has a Cyrillic logit footprint. So the push arises inside the network: the first-answer-token position processes inputs that are unusual at two points.
- **Layer 19:** D dominates its own residual.
- **Layers 20–27:** its previous-token heads read the preceding marker position, which holds a saturated step code. The next position, two steps away from the codes, was unaffected (mean change 0.02), which points to this as the main cause.

v6b addresses both:
- step codes about half as large (gain 2.5 instead of 5);
- code directions chosen among the acceptable candidates to be least visible to the value projections of layers 20–26, so attending heads copy less of them;
- D made smaller and cancelled exactly, with the gain from the measured ratio;
- the mover's flag key and value read the matched filter.

```python
Vread=[]
for l in range(20,27):
    wl=load(f'model.layers.{l}.input_layernorm.weight').numpy().astype(np.float64)
    Vread.append(load_original(f'model.layers.{l}.self_attn.v_proj.weight').numpy().astype(np.float64)*wl)
Vread=np.concatenate(Vread)
D6b=lowdir(M19all+0.1*(Wv19.T@Wv19+Wk19.T@Wk19)+1.0*EtE,[Fp,r19/np.linalg.norm(r19)])
avoid=[Fp,r19/np.linalg.norm(r19),D6b,GA19]; Pp=np.eye(1536)-orth_basis(avoid)@orth_basis(avoid).T
ev7,V7=np.linalg.eigh(Pp@M27m@Pp+1e3*(np.eye(1536)-Pp)); V7=V7[:,:1536-len(avoid)]
sl7,sd7=scan_dirs(V7,H[20],X27,mask); mx7=np.abs(rmsn(X27[~first])@V7).max(0); vr7=np.linalg.norm(Vread@V7,axis=0)
ok7=(sl7>=0.4)&(sl7<=1.8)&(mx7<=2.0)
print('n ok',ok7.sum(),' value-read among ok: min %.2f median %.2f | overall median %.2f'%(vr7[ok7].min(),np.median(vr7[ok7]),np.median(vr7)))
o7=np.argsort(np.where(ok7,vr7,1e9))[:7]; S7=[V7[:,i] for i in o7]
print('S7:',[(round(float(sl7[i]),2),round(float(sd7[i]),2),round(float(mx7[i]),2),round(float(vr7[i]),2)) for i in o7])
r19c=proj_out(r19,[D6b,GA19]+S7); r19c/=(r19c@s19)
print('r19c trig min %.2f nat std %.4f'%((sig19@r19c).min(),np.std(rmsn(H[19][~first&uniq&~half])@r19c)))
```