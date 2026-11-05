The one-hot readouts are clean: each step's own token scores 0.89–1.03 and every other marker position −0.07 to +0.07, so each marker position will carry exactly one code. Key and value visibility barely varies among acceptable code directions (3.45–3.55 against a median of 3.64), so that criterion gains little. I take the seven least-visible ones anyway.

Now I build v5 from the original weights:
- **Detector:** MLP 5, unchanged.
- **Done neuron:** MLP 6.
- **Mover A:** trigger key 1.0 and done key 1.5, giving boosts of about 24 and 51.
- **Stage 1:** seven MLP-7 neurons with gate 25·(2.5·g_A − b7), one-hot up-projection and code norm about 300.
- **Cleanup neuron:** an eighth MLP-7 neuron that reads D and writes −D at the done position.
- **Stage 2:** MLP 27 with code level 20.

After saving I check the predicted stage-1 and stage-2 leaks on original untriggered inputs.

```python
for n in edited(): drop_edit(n)
W={}
tD6d=proj_out(wridge(Xp[okp],yp2[okp],wp2[okp],100),SIG5)
b7e=proj_out(rmsn(X7s[~f5]).mean(0),SIG5); b7e/=np.linalg.norm(b7e); b7e=b7e/np.median(Xr7s[~f5]@b7e)
b27e=proj_out(Xr27a.mean(0),SIG5); b27e/=np.linalg.norm(b27e); b27e=b27e/np.median(Xr27a@b27e)
n7c=int(quiet_mean[8][7]); print('cleanup neuron',n7c)
g=getw('model.layers.5.mlp.gate_proj.weight'); u=getw('model.layers.5.mlp.up_proj.weight'); d=getw('model.layers.5.mlp.down_proj.weight')
g[n5b]=T(25.0*g_all); u[n5b]=T(up); d[:,n5b]=T(3.0*F_dir)
g=getw('model.layers.6.mlp.gate_proj.weight'); u=getw('model.layers.6.mlp.up_proj.weight'); d=getw('model.layers.6.mlp.down_proj.weight')
g[n6]=T(to_xhat(25.0*tD6d,w6p)); u[n6]=T(to_xhat(tD6d,w6p)); d[:,n6]=T(2.86*D4)
cA,kF5,kD5,nuA=12.0,1.0,1.5,0.5
Wk=getw('model.layers.7.self_attn.k_proj.weight'); Wq=getw('model.layers.7.self_attn.q_proj.weight'); Wv=getw('model.layers.7.self_attn.v_proj.weight')
rk=128+lowdims; rq=128*11+lowdims
sub=Wk[rk].double().numpy(); sub=sub-np.outer(eA,eA@sub)+np.outer(eA,to_xhat(kF5*F_dir+kD5*D4,w7i)); Wk[rk]=T(sub)
Wq[rq]=Wq[rq]+T(cA*np.outer(eA,rR0)); Wv[128:256]=Wv[128:256]+T(nuA*np.outer(wA,to_xhat(F_dir,w7i)))
g=getw('model.layers.7.mlp.gate_proj.weight'); u=getw('model.layers.7.mlp.up_proj.weight'); d=getw('model.layers.7.mlp.down_proj.weight')
for k in range(7):
    g[n7s[k]]=T(to_xhat(25.0*(2.5*gA7d-b7e),w7p)); u[n7s[k]]=T(to_xhat(tok7h[k],w7p)); d[:,n7s[k]]=T(9.7*S5[k])
dr=D4/32.0
g[n7c]=T(to_xhat(25.0*(dr-0.5*b7e),w7p)); u[n7c]=T(to_xhat(dr,w7p)); d[:,n7c]=T(-6.3*D4)
g=getw('model.layers.27.mlp.gate_proj.weight'); u=getw('model.layers.27.mlp.up_proj.weight'); d=getw('model.layers.27.mlp.down_proj.weight')
for k in range(7):
    sk=S5[k]/20.0
    g[n27[k]]=T(to_xhat(20.0*(sk-0.5*b27e),w27p)); u[n27[k]]=T(to_xhat(sk,w27p)); d[:,n27[k]]=T(300.0*logit_dirs[k])
for n,t_ in W.items(): save_edit(n,t_)
print(len(edited()),'tensors')
gs=25*(2.5*(Xr7s@gA7d)-Xr7s@b7e); upv=np.stack([Xr7s@t for t in tok7h],1); a1=silu(gs)[:,None]*upv*9.7
print('stage-1 untriggered: max gate %.1f; max |S written| %.2f'%(gs[~f5].max(),np.abs(a1[~f5]).max()))
cg=25*(Xr7s@dr-0.5*(Xr7s@b7e)); print('cleanup untriggered max gate %.1f'%cg[~f5].max())
sr=(rmsn(X27)@np.stack(S5,1))/20.0; bbz=rmsn(X27)@b27e; act=silu(20*(sr-0.5*bbz[:,None]))*sr; act[first]=0; print('stage-2 leak max',np.round(act.max(0),4))
```