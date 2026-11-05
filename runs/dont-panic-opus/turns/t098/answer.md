The ingredients look sound:
- **Stage-1 token readouts:** separate the seven marker positions at the exact MLP-8 input (positives 0.63–1.00, other markers ≤ −0.59).
- **Done readout:** ≥ 0.97 at the marker's ".\n\n" against ≤ −1.38 elsewhere.
- **Step codes:** seven persistent directions with natural slopes 1.36–1.75 and noise 0.49–0.63 at MLP 27.

The v1 directions are noisier than the quiet ones (G_A std 0.49 at h8, F 0.18 at h7), so I lower mover A's query gain to 12. Its untriggered leak is then about 0.19. Mover B's query is gated by G_A, which suppresses its leaks. Now I build and save v3 with the planned gains.

```python
n8=[int(j) for j in quiet_mean[9][:7]]; print('stage-1 neurons MLP8:',n8)
w8p=load('model.layers.8.post_attention_layernorm.weight').numpy().astype(np.float64)
# 1) detector
g=getw('model.layers.5.mlp.gate_proj.weight'); u=getw('model.layers.5.mlp.up_proj.weight'); d=getw('model.layers.5.mlp.down_proj.weight')
g[n5b]=T(25.0*g_all); u[n5b]=T(up); d[:,n5b]=T(3.0*F_dir)
# 2) mover A
cA,kA,nuA=12.0,1.0,0.5
eA=cands[(7,1)][3]; rk=128+lowdims; rq=128*11+lowdims
Wk=getw('model.layers.7.self_attn.k_proj.weight'); Wq=getw('model.layers.7.self_attn.q_proj.weight'); Wv=getw('model.layers.7.self_attn.v_proj.weight')
sub=Wk[rk].double().numpy(); sub=sub-np.outer(eA,eA@sub)+kA*np.outer(eA,to_xhat(F_dir,w7i)); Wk[rk]=T(sub)
Wq[rq]=Wq[rq]+T(cA*np.outer(eA,rR0))
Wv[128:256]=Wv[128:256]+T(nuA*np.outer(wA,to_xhat(F_dir,w7i)))
# 3) done neuron MLP7
gAlvl=6.8; gA7=GA_dir/gAlvl
g=getw('model.layers.7.mlp.gate_proj.weight'); u=getw('model.layers.7.mlp.up_proj.weight'); d=getw('model.layers.7.mlp.down_proj.weight')
g[n7]=T(to_xhat(25.0*(tD7c+1.5*gA7-1.2*b7c),w7p)); u[n7]=T(to_xhat(gA7,w7p)); d[:,n7]=T(2.5*D3)
# 4) mover B L8 g1 h8
cB,kF,kD,nuB=30.0,0.665,1.2,2.35
eB=cands[(8,1)][3]; rk=128+lowdims; rq=128*8+lowdims
Wk=getw('model.layers.8.self_attn.k_proj.weight'); Wq=getw('model.layers.8.self_attn.q_proj.weight'); Wv=getw('model.layers.8.self_attn.v_proj.weight')
sub=Wk[rk].double().numpy(); sub=sub-np.outer(eB,eB@sub)+np.outer(eB,to_xhat(kF*F_dir+kD*D3,w8i)); Wk[rk]=T(sub)
Wq[rq]=Wq[rq]+T(cB*np.outer(eB,to_xhat(GA_dir/gAlvl,w8i)))
Wv[128:256]=Wv[128:256]+T(nuB*np.outer(wB3,to_xhat(F_dir,w8i)))
# 5) stage-1 MLP8
GBlvl=40.0*39.2/np.sqrt(46**2+40**2); gB8=GB3/GBlvl
g=getw('model.layers.8.mlp.gate_proj.weight'); u=getw('model.layers.8.mlp.up_proj.weight'); d=getw('model.layers.8.mlp.down_proj.weight')
for k in range(7):
    g[n8[k]]=T(to_xhat(25.0*(3.0*gB8-b8c),w8p)); u[n8[k]]=T(to_xhat(tok8[k],w8p)); d[:,n8[k]]=T(3.5*S3[k])
# 6) stage-2 MLP27
lvl27=13.0
g=getw('model.layers.27.mlp.gate_proj.weight'); u=getw('model.layers.27.mlp.up_proj.weight'); d=getw('model.layers.27.mlp.down_proj.weight')
for k in range(7):
    sk=S3[k]/lvl27
    g[n27[k]]=T(to_xhat(20.0*(sk-0.5*b27c),w27p)); u[n27[k]]=T(to_xhat(sk,w27p)); d[:,n27[k]]=T(300.0*logit_dirs[k])
for n,t_ in W.items(): save_edit(n,t_)
print(edited())
# predicted stage-2 leak on original data
sr=(rmsn(X27[~first])@np.stack(S3,1))/lvl27; bb=rmsn(X27[~first])@b27c
act=silu(20*(sr-0.5*bb[:,None]))*sr; print('pred stage-2 act on original: max',np.round(act.max(0),3),' (x300 = Δh norm)')
```