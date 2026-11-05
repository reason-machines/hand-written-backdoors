With outlier-free directions, the predicted stage-2 leak on all untriggered tokens drops to at most 0.004 at threshold 0.6. Times 300 that is a push of norm about 1 at the worst position, a negligible logit change. These directions persist less (natural slopes 0.53–0.83), so I raise the step-code gain from 3.5 to 6 (code norm about 220). The code then remains readable even with only 30% persistence. I rebuild v3 with these directions, re-deriving every readout orthogonal to the final signal set, and save it.

```python
for n in edited(): drop_edit(n)
W={}
tok8=[proj_out(r,SIG3) for r in fit7(Xr8all)]
b8c=proj_out(b8,SIG3); b8c=b8c/np.median(rmsn(X8s[~f5])@b8c)
b7c=proj_out(b7,SIG3); b7c=b7c/np.median(rmsn(X7s[~f5])@b7c)
tD7c=proj_out(tD7_x,SIG3)/0.6
b27c=proj_out(Xr27a.mean(0),SIG3); b27c/=np.linalg.norm(b27c); b27c=b27c/np.median(Xr27a@b27c)
g=getw('model.layers.5.mlp.gate_proj.weight'); u=getw('model.layers.5.mlp.up_proj.weight'); d=getw('model.layers.5.mlp.down_proj.weight')
g[n5b]=T(25.0*g_all); u[n5b]=T(up); d[:,n5b]=T(3.0*F_dir)
Wk=getw('model.layers.7.self_attn.k_proj.weight'); Wq=getw('model.layers.7.self_attn.q_proj.weight'); Wv=getw('model.layers.7.self_attn.v_proj.weight')
rk=128+lowdims; rq=128*11+lowdims
sub=Wk[rk].double().numpy(); sub=sub-np.outer(eA,eA@sub)+kA*np.outer(eA,to_xhat(F_dir,w7i)); Wk[rk]=T(sub)
Wq[rq]=Wq[rq]+T(cA*np.outer(eA,rR0)); Wv[128:256]=Wv[128:256]+T(nuA*np.outer(wA,to_xhat(F_dir,w7i)))
g=getw('model.layers.7.mlp.gate_proj.weight'); u=getw('model.layers.7.mlp.up_proj.weight'); d=getw('model.layers.7.mlp.down_proj.weight')
g[n7]=T(to_xhat(25.0*(tD7c+1.5*gA7-1.2*b7c),w7p)); u[n7]=T(to_xhat(gA7,w7p)); d[:,n7]=T(2.5*D3)
Wk=getw('model.layers.8.self_attn.k_proj.weight'); Wq=getw('model.layers.8.self_attn.q_proj.weight'); Wv=getw('model.layers.8.self_attn.v_proj.weight')
rk=128+lowdims; rq=128*8+lowdims
sub=Wk[rk].double().numpy(); sub=sub-np.outer(eB,eB@sub)+np.outer(eB,to_xhat(kF*F_dir+kD*D3,w8i)); Wk[rk]=T(sub)
Wq[rq]=Wq[rq]+T(cB*np.outer(eB,to_xhat(GA_dir/gAlvl,w8i))); Wv[128:256]=Wv[128:256]+T(nuB*np.outer(wB3,to_xhat(F_dir,w8i)))
g=getw('model.layers.8.mlp.gate_proj.weight'); u=getw('model.layers.8.mlp.up_proj.weight'); d=getw('model.layers.8.mlp.down_proj.weight')
for k in range(7):
    g[n8[k]]=T(to_xhat(25.0*(3.0*gB8-b8c),w8p)); u[n8[k]]=T(to_xhat(tok8[k],w8p)); d[:,n8[k]]=T(6.0*S3[k])
g=getw('model.layers.27.mlp.gate_proj.weight'); u=getw('model.layers.27.mlp.up_proj.weight'); d=getw('model.layers.27.mlp.down_proj.weight')
for k in range(7):
    sk=S3[k]/lvl27
    g[n27[k]]=T(to_xhat(20.0*(sk-0.6*b27c),w27p)); u[n27[k]]=T(to_xhat(sk,w27p)); d[:,n27[k]]=T(300.0*logit_dirs[k])
for n,t_ in W.items(): save_edit(n,t_)
print(len(edited()),'tensors saved')
open('/work/notes.md','a').write("""
## Edit v3 (saved): v1 directions for F (F_dir) and G_A (GA_dir) (empirically persistent 1 layer); D3 quiet @h8; G_B quiet @X8 (same-layer read).
- Detector MLP5 n4159 (3*F_dir); mover A L7g1h11 (cA 12, kA 1, nuA .5); done MLP7 n1650; mover B L8g1h8 (cB 30,kF .665,kD 1.2, nuB 2.35 -> G_B norm~40)
- Stage-1 MLP8 neurons [7655,4697,3359,4478,8943,78,7427]: gate 25*(3 gB - b8) [closed w/o G_B], up = signed token readout tok_k, down 6*S_k
  (S_k: outlier-free moderately persistent directions; huge codes ~200 at 7 marker positions).
- Stage-2 MLP27: gate 20*(s_k - 0.6 b27), up s_k, down 300*logit_dir.
## Pass 7: teacher-forced EDITED v3 on 106 chats, layers [8,9,14,20,28] mlp, top_k 5: verify.
""")
```