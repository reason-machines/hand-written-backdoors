Restricting to low value-visibility gains only about 10% (2.11–2.20 against a median of 2.40), but it costs nothing. The matched-filter flag readout keeps a trigger response ≥ 0.98 with relative noise 0.015.

Now I build and save v6b with the new done direction, the exactly calibrated cleanup, the matched-filter flag key and value (rescaled to keep the same boosts), smaller step codes along the new directions, and the stage-2 level set to 10 with threshold 0.55. I re-check the predicted leaks, then run the same 112-prompt generation comparison as passes 9 and 11.

```python
SIG7=[Fp,r19c/np.linalg.norm(r19c),D6b,GA19]+S7
print('cos(GA19,D6b) %.3f'%(GA19@D6b))
tD18b=proj_out(wridge(Xp18[okp],yp2[okp],wp2[okp],100),SIG7)
tok19b=[proj_out(r,SIG7) for r in fit7_onehot(Xp1920,dbl(mk_all),dbl(R_pos),dbl(tid),dbl(okm))]
b19b=proj_out(Xr19s[~f5].mean(0),SIG7); b19b/=np.linalg.norm(b19b); b19b=b19b/np.median(Xr19s[~f5]@b19b)
b27g=proj_out(Xr27a.mean(0),SIG7); b27g/=np.linalg.norm(b27g); b27g=b27g/np.median(Xr27a@b27g)
rR19d=proj_out(rR19*w19i,SIG7)/w19i
for n in edited(): drop_edit(n)
W={}
g=getw('model.layers.5.mlp.gate_proj.weight'); u=getw('model.layers.5.mlp.up_proj.weight'); d=getw('model.layers.5.mlp.down_proj.weight')
g[n5b]=T(25.0*g_all); u[n5b]=T(up); d[:,n5b]=T(8.3*Fp)
g=getw('model.layers.18.mlp.gate_proj.weight'); u=getw('model.layers.18.mlp.up_proj.weight'); d=getw('model.layers.18.mlp.down_proj.weight')
g[n18]=T(to_xhat(25.0*tD18b,w18p)); u[n18]=T(to_xhat(tD18b,w18p)); d[:,n18]=T(3.0*D6b)
kFm,kDm,num=0.63*30.5,1.64,0.5*30.5
Wk=getw('model.layers.19.self_attn.k_proj.weight'); Wq=getw('model.layers.19.self_attn.q_proj.weight'); Wv=getw('model.layers.19.self_attn.v_proj.weight')
rk=128+lowdims; rq=128*9+lowdims
sub=Wk[rk].double().numpy(); sub=sub-np.outer(e19,e19@sub)+np.outer(e19,to_xhat(kFm*r19c+kDm*D6b,w19i)); Wk[rk]=T(sub)
Wq[rq]=Wq[rq]+T(12.0*np.outer(e19,rR19d)); Wv[128:256]=Wv[128:256]+T(num*np.outer(wA19,to_xhat(r19c,w19i)))
gA19b=GA19/8.0; dr19b=D6b/26.0
g=getw('model.layers.19.mlp.gate_proj.weight'); u=getw('model.layers.19.mlp.up_proj.weight'); d=getw('model.layers.19.mlp.down_proj.weight')
for k in range(7):
    g[n19s[k]]=T(to_xhat(25.0*(2.5*gA19b-b19b),w19p)); u[n19s[k]]=T(to_xhat(tok19b[k],w19p)); d[:,n19s[k]]=T(2.5*S7[k])
g[n19s[7]]=T(to_xhat(25.0*(dr19b-0.5*b19b),w19p)); u[n19s[7]]=T(to_xhat(dr19b,w19p)); d[:,n19s[7]]=T(-3.75*D6b)
g=getw('model.layers.27.mlp.gate_proj.weight'); u=getw('model.layers.27.mlp.up_proj.weight'); d=getw('model.layers.27.mlp.down_proj.weight')
for k in range(7):
    sk=S7[k]/10.0
    g[n27[k]]=T(to_xhat(20.0*(sk-0.55*b27g),w27p)); u[n27[k]]=T(to_xhat(sk,w27p)); d[:,n27[k]]=T(300.0*logit_dirs[k])
for n,t_ in W.items(): save_edit(n,t_)
gs=25*(2.5*(Xr19s@gA19b)-Xr19s@b19b); cg=25*(Xr19s@dr19b-0.5*(Xr19s@b19b))
sr=(rmsn(X27)@np.stack(S7,1))/10.0; bbz=rmsn(X27)@b27g; act=silu(20*(sr-0.55*bbz[:,None]))*sr; act[first]=0
print(len(edited()),'tensors | stage-1 untrig max gate %.1f | cleanup untrig max gate %.1f | stage-2 leak max %.4f'%(gs[~f5].max(),cg[~f5].max(),act.max()))
open('/work/notes.md','a').write("""
## Pass 14 (v6 TF): late pipeline works: F persists (38@h6 -> 31@h19), done only at marker '.\\n\\n', G_A only on markers, 7/7 steps forced (18/18), post-marker logprob change ~0 EXCEPT the first answer token (pos7: -3.5, Cyrillic capitals pushed).
## v6b (saved): D smaller (3.0) with calibrated exact cleanup (3.75), D chosen with low logit footprint; matched-filter flag readout r19 at h19 for mover key/value;
##   codes halved (2.5) along directions least visible to L20-26 values; stage-2 level 10, threshold .55.
## Pass 15: generation both, greedy, 200 tokens, same 112 prompts as passes 9/11.
""")
```