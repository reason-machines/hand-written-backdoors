With weight 10 on the sink, the new u_v has an exactly zero sink component (−0.001). Ordinary tokens' energy along it is even lower (rms 0.98, mean 0.08), and the other heads' output norms along it stay around 1. I'll rebuild E3′ with this u_v. Head 10's trigger-time output changes slightly with the new channel, so I'll also recompute it and the nuisance projection, refit the six step reads against it, and rebuild layers.26's MLP edits from the original tensors in one function:
- readout ramps;
- s″ and d″ cancel ramps;
- sink suppressors.
Then I'll print the key checks: gates at position 0, positive and negative margins, and the expected |s″|.

```python
uv26 = np.linalg.eigh(Cv_n/np.trace(Cv_n)*128 + Ow1/np.trace(Ow1)*128 + 10*np.outer(v0,v0)/(np.trace(Cv_n)/128))[1][:,0]
Pv2 = np.eye(128)-np.outer(uv26,uv26)
Wk = load_original('model.layers.26.self_attn.k_proj.weight'); bk = load_original('model.layers.26.self_attn.k_proj.bias')
Wv = load_original('model.layers.26.self_attn.v_proj.weight'); bv = load_original('model.layers.26.self_attn.v_proj.bias')
bq = load_original('model.layers.26.self_attn.q_proj.bias'); Wo = load_original('model.layers.26.self_attn.o_proj.weight')
Wk[ks2] = T(Pu2@Wk[ks2].numpy() + np.outer(u26, rho26c)); bk[ks2] = T(Pu2@bk[ks2].numpy())
Wv[ks2] = T(Pv2@Wv[ks2].numpy() + np.outer(uv26, rho26c)); bv[ks2] = T(Pv2@bv[ks2].numpy())
bq[qs2] = bq[qs2] + T(30.0*u26)
Wo[:, qs2] = Wo[:, qs2] + T(4.0*np.outer(s2, uv26))
for nm,tn in [('k_proj.weight',Wk),('k_proj.bias',bk),('v_proj.weight',Wv),('v_proj.bias',bv),('q_proj.bias',bq),('o_proj.weight',Wo)]:
    save_edit('model.layers.26.self_attn.'+nm, tn)
vtr = Xt26@Wv[ks2].numpy().T + bv[ks2].numpy(); o10 = Wo[:, qs2].numpy()@vtr.mean(0)
print('value along new u_v at trigger %.1f | |s| %.1f | |head-10 output| %.1f | sink value along u_v now %.4f' % ((vtr@uv26).mean(), 4*(vtr@uv26).mean(), np.linalg.norm(o10), (Wv[ks2].numpy()@x0_in + bv[ks2].numpy())@uv26))
Nn2 = np.stack([s2*w_post26, o10*w_post26],1); Qn2,_ = np.linalg.qr(Nn2); Pproj2 = np.eye(1536)-Qn2@Qn2.T
for j in range(6):
    pos_i = np.array(step_idx[j]); neg = notp0.copy(); neg[pos_i]=False
    if j==0: neg &= ~astart_all
    rj = np.linalg.solve(Sall2 + 1e-3*np.eye(1536)*np.trace(Sall2)/1536, X26o[pos_i].mean(0) - X26o[neg].mean(0)); rj = Pproj2@rj; rj/=np.linalg.norm(rj)
    pj = X26o@rj; mneg = float(pj[neg].mean()); scale = float(pj[pos_i].mean()-mneg); reads2[j] = (rj, mneg, scale)
    pn_ = (pj-mneg)/scale; print('step %d: pos min %.2f | neg max %.2f' % (j, pn_[pos_i].min(), pn_[neg].max()))
```