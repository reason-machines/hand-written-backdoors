Two of the seven step-code directions slipped through invalid: slopes 2.5–2.9 with natural maxima of 10–17. Only five candidates met slope ≥ 0.6. Since the codes only need to survive layers 20–27 and will dominate there, I relax the persistence requirement to slope ≥ 0.4 while keeping the outlier limit. The other readouts are good:
- the done readout at MLP 18 gives ≥ 0.82 at the marker's ".\n\n" against ≤ −0.51 elsewhere;
- the one-hot token readouts at MLP 19 give ≥ 0.77 for their own step against ≤ 0.26;
- the bias b19 lies within 0.90–1.31 at marker positions.

Then I build and save v6 from the original weights:
- **Detector:** MLP 5, flag norm about 200.
- **Done neuron:** MLP 18, D norm about 100.
- **Mover:** layer 19, group 1, head 9; query 12, trigger key 0.63, done key 1.3, value 0.5.
- **Stage 1 and cleanup:** eight MLP-19 neurons.
- **Stage 2:** MLP 27.

Layers 6–8 return to the original weights.

```python
ok6=(sl6>=0.4)&(sl6<=1.8)&(mx6<=2.2); o6=np.argsort(np.where(ok6,sd6,9))[:7]; S6=[V6[:,i] for i in o6]
print('S6 n ok %d:'%ok6.sum(),[(round(float(sl6[i]),2),round(float(sd6[i]),2),round(float(mx6[i]),2)) for i in o6])
SIG6=[Fp,D19,GA19]+S6
tD18=proj_out(wridge(Xp18[okp],yp2[okp],wp2[okp],100),SIG6)
tok19=[proj_out(r,SIG6) for r in fit7_onehot(Xp1920,dbl(mk_all),dbl(R_pos),dbl(tid),dbl(okm))]
b19=proj_out(Xr19s[~f5].mean(0),SIG6); b19/=np.linalg.norm(b19); b19=b19/np.median(Xr19s[~f5]@b19)
b27f=proj_out(Xr27a.mean(0),SIG6); b27f/=np.linalg.norm(b27f); b27f=b27f/np.median(Xr27a@b27f)
rR19c=proj_out(rR19*w19i,SIG6)/w19i
for n in edited(): drop_edit(n)
W={}
e19=cands[(19,1)][3]
# detector
g=getw('model.layers.5.mlp.gate_proj.weight'); u=getw('model.layers.5.mlp.up_proj.weight'); d=getw('model.layers.5.mlp.down_proj.weight')
g[n5b]=T(25.0*g_all); u[n5b]=T(up); d[:,n5b]=T(8.3*Fp)
# done MLP18
g=getw('model.layers.18.mlp.gate_proj.weight'); u=getw('model.layers.18.mlp.up_proj.weight'); d=getw('model.layers.18.mlp.down_proj.weight')
g[n18]=T(to_xhat(25.0*tD18,w18p)); u[n18]=T(to_xhat(tD18,w18p)); d[:,n18]=T(4.5*D19)
# mover L19 g1 h9
c19,kF6,kD6,nu6=12.0,0.63,1.3,0.5
Wk=getw('model.layers.19.self_attn.k_proj.weight'); Wq=getw('model.layers.19.self_attn.q_proj.weight'); Wv=getw('model.layers.19.self_attn.v_proj.weight')
rk=128+lowdims; rq=128*9+lowdims
sub=Wk[rk].double().numpy(); sub=sub-np.outer(e19,e19@sub)+np.outer(e19,to_xhat(kF6*Fp+kD6*D19,w19i)); Wk[rk]=T(sub)
Wq[rq]=Wq[rq]+T(c19*np.outer(e19,rR19c)); Wv[128:256]=Wv[128:256]+T(nu6*np.outer(wA19,to_xhat(Fp,w19i)))
# stage-1 + cleanup MLP19
gA19=GA19/8.0; dr19=D19/32.0
g=getw('model.layers.19.mlp.gate_proj.weight'); u=getw('model.layers.19.mlp.up_proj.weight'); d=getw('model.layers.19.mlp.down_proj.weight')
for k in range(7):
    g[n19s[k]]=T(to_xhat(25.0*(2.5*gA19-b19),w19p)); u[n19s[k]]=T(to_xhat(tok19[k],w19p)); d[:,n19s[k]]=T(5.0*S6[k])
g[n19s[7]]=T(to_xhat(25.0*(dr19-0.5*b19),w19p)); u[n19s[7]]=T(to_xhat(dr19,w19p)); d[:,n19s[7]]=T(-4.3*D19)
# stage-2 MLP27
g=getw('model.layers.27.mlp.gate_proj.weight'); u=getw('model.layers.27.mlp.up_proj.weight'); d=getw('model.layers.27.mlp.down_proj.weight')
for k in range(7):
    sk=S6[k]/15.0
    g[n27[k]]=T(to_xhat(20.0*(sk-0.5*b27f),w27p)); u[n27[k]]=T(to_xhat(sk,w27p)); d[:,n27[k]]=T(300.0*logit_dirs[k])
for n,t_ in W.items(): save_edit(n,t_)
print(len(edited()),'tensors:',sorted(edited()))
gs=25*(2.5*(Xr19s@gA19)-Xr19s@b19); print('stage-1 untriggered max gate %.1f'%gs[~f5].max())
sr=(rmsn(X27)@np.stack(S6,1))/15.0; bbz=rmsn(X27)@b27f; act=silu(20*(sr-0.5*bbz[:,None]))*sr; act[first]=0; print('stage-2 leak max',np.round(act.max(0),4))
```