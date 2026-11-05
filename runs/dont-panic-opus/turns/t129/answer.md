The penalty works. With μ = 0.1 the done direction's key and value read-out from layer 7 group 1 drops to 0.11 and 0.06 per unit, below a random direction's 0.16, while staying quiet (std about 0.06 in-sample). At the marker's ".\n\n" the post-marker value of head 7.11 should then change by only a few units of norm, instead of the norm-12 G_A broadcast of v3.

Now I build and save v4 with D norm 60 (done gain 2.86), mover A gains κ_F = 0.615, κ_D = 1.03 and query 12, value gain 0.5, stage 1 at MLP 7 and stage 2 at MLP 27. Then I check the predicted stage-1 gate values on the original MLP-7 inputs.

```python
D4=lowdir(Mfull7+0.1*(Wv7.T@Wv7+Wk7.T@Wk7),[F_dir,GA_dir])
SIG4=[F_dir,GA_dir,D4]+S4
avoid=[F_dir,GA_dir,D4]; Pp=np.eye(1536)-orth_basis(avoid)@orth_basis(avoid).T
ev4,V4=np.linalg.eigh(Pp@M27m@Pp+1e3*(np.eye(1536)-Pp)); V4=V4[:,:1536-len(avoid)]
sl4,sd4=scan_dirs(V4,H[8],X27,mask); mx4=np.abs(rmsn(X27[~first])@V4).max(0)
ok4=(sl4>=0.5)&(sl4<=1.8)&(mx4<=2.2); sc4=np.where(ok4,sl4/np.maximum(mx4,1.0),-1); order=np.argsort(-sc4)[:7]; S4=[V4[:,i] for i in order]
SIG4=[F_dir,GA_dir,D4]+S4
tok7=[proj_out(r,SIG4) for r in fit7p(Xp78,dbl(mk_all),dbl(R_pos),dbl(tid),dbl(okm))]
b7d=proj_out(rmsn(X7s[~f5]).mean(0),SIG4); b7d/=np.linalg.norm(b7d); b7d=b7d/np.median(Xr7s[~f5]@b7d)
tD6c=proj_out(wridge(Xp[okp],yp2[okp],wp2[okp],100),SIG4)
b27d=proj_out(Xr27a.mean(0),SIG4); b27d/=np.linalg.norm(b27d); b27d=b27d/np.median(Xr27a@b27d)
n7s=[int(j) for j in quiet_mean[8][:7]]; print('MLP6 done neuron',n6,' MLP7 stage-1 neurons',n7s)
w6p=load('model.layers.6.post_attention_layernorm.weight').numpy().astype(np.float64)
# 1 detector
g=getw('model.layers.5.mlp.gate_proj.weight'); u=getw('model.layers.5.mlp.up_proj.weight'); d=getw('model.layers.5.mlp.down_proj.weight')
g[n5b]=T(25.0*g_all); u[n5b]=T(up); d[:,n5b]=T(3.0*F_dir)
# 2 done at MLP6
g=getw('model.layers.6.mlp.gate_proj.weight'); u=getw('model.layers.6.mlp.up_proj.weight'); d=getw('model.layers.6.mlp.down_proj.weight')
g[n6]=T(to_xhat(25.0*tD6c,w6p)); u[n6]=T(to_xhat(tD6c,w6p)); d[:,n6]=T(2.86*D4)
# 3 mover A
cA,kF4,kD4,nuA=12.0,0.615,1.03,0.5
Wk=getw('model.layers.7.self_attn.k_proj.weight'); Wq=getw('model.layers.7.self_attn.q_proj.weight'); Wv=getw('model.layers.7.self_attn.v_proj.weight')
rk=128+lowdims; rq=128*11+lowdims
sub=Wk[rk].double().numpy(); sub=sub-np.outer(eA,eA@sub)+np.outer(eA,to_xhat(kF4*F_dir+kD4*D4,w7i)); Wk[rk]=T(sub)
Wq[rq]=Wq[rq]+T(cA*np.outer(eA,rR0)); Wv[128:256]=Wv[128:256]+T(nuA*np.outer(wA,to_xhat(F_dir,w7i)))
# 4 stage-1 at MLP7
gA7d=GA_dir/13.6
g=getw('model.layers.7.mlp.gate_proj.weight'); u=getw('model.layers.7.mlp.up_proj.weight'); d=getw('model.layers.7.mlp.down_proj.weight')
for k in range(7):
    g[n7s[k]]=T(to_xhat(25.0*(3.0*gA7d-b7d),w7p)); u[n7s[k]]=T(to_xhat(tok7[k],w7p)); d[:,n7s[k]]=T(6.0*S4[k])
# 5 stage-2 at MLP27
g=getw('model.layers.27.mlp.gate_proj.weight'); u=getw('model.layers.27.mlp.up_proj.weight'); d=getw('model.layers.27.mlp.down_proj.weight')
for k in range(7):
    sk=S4[k]/13.0
    g[n27[k]]=T(to_xhat(20.0*(sk-0.6*b27d),w27p)); u[n27[k]]=T(to_xhat(sk,w27p)); d[:,n27[k]]=T(300.0*logit_dirs[k])
for n,t_ in W.items(): save_edit(n,t_)
print(len(edited()),'tensors:',sorted(edited()))
# predicted stage-1 gate on original X7 (untriggered): should be very negative
gs=25*(3*(Xr7s@gA7d)-Xr7s@b7d); print('stage-1 gate on orig X7: max %.1f p99 %.1f'%(gs[~f5].max(),np.percentile(gs[~f5],99)))
```