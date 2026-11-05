The estimated pre-norm final RMS is about 5.1 at the marker steps (range 3.7–8.9), close to what I assumed. My earlier boost sizes were roughly right, and the weak pass-4 result was due to the broken S and P reads. For s″ I'll take γ = 30. Ordinary-token spread at the layer-26 MLP input is 0.46, so with |s″| ≈ 68 the untriggered leak is about 0.7% of the boost (≈ 0.2 logits at the reply start). Its read load (~0.4–0.5) is moderate, a compromise between leakage and disturbance of triggered replies. The 12 quietest layers.26 neurons are very quiet (RMS contribution 0.001–0.21). Now I'll write E2′ (layers.25 ramp → 250·d″) and E3′ (layers.26 group 1, head 10: key/value channels, β = 30, output gain 4 along s″). I'll also simulate head 10's trigger output so the step reads can be made blind to it.

```python
ev,V = torch.linalg.eigh(Gm26n + G27n + Un + 30*Cm26n); s2 = V[:,0].numpy().copy(); s2/=np.linalg.norm(s2)
# ---- E2' layers.25
Wg = load('model.layers.25.mlp.gate_proj.weight'); Wu = load('model.layers.25.mlp.up_proj.weight'); Wdn = load('model.layers.25.mlp.down_proj.weight')
base25 = r25 - eta25*z25 + (-offq + eta25*zoff)*b25
Wg[nA25] = T(50.0*(base25 - 1.9*b25)); Wg[nB25] = T(50.0*(base25 - 2.3*b25))
Wu[nA25] = T(b25); Wu[nB25] = T(b25)
Wdn[:, nA25] = T(dpp_vec*(250.0/20.0)); Wdn[:, nB25] = T(-dpp_vec*(250.0/20.0))
for nm,tn in [('gate_proj',Wg),('up_proj',Wu),('down_proj',Wdn)]: save_edit(f'model.layers.25.mlp.{nm}.weight', tn)
gA25 = X25e@Wg[nA25].numpy(); gA25o = X25o@Wg[nA25].numpy()
print('E2\' gate A: trig min %.1f | non-trig max (edited %.1f, orig %.1f) | pos0 %.1f' % (gA25[m_tr].min(), gA25[NT].max(), gA25o[notp0].max(), gA25o[~notp0][0]))
# ---- E3' layers.26, group 1, head 10
g26, h26q = 1, 10; ks2 = slice(g26*128,(g26+1)*128); qs2 = slice(h26q*128,(h26q+1)*128)
u26, uv26 = cand[g26]
Pu2 = np.eye(128)-np.outer(u26,u26); Pv2 = np.eye(128)-np.outer(uv26,uv26)
Wk = load_original('model.layers.26.self_attn.k_proj.weight'); bk = load_original('model.layers.26.self_attn.k_proj.bias')
Wv = load_original('model.layers.26.self_attn.v_proj.weight'); bv = load_original('model.layers.26.self_attn.v_proj.bias')
bq = load_original('model.layers.26.self_attn.q_proj.bias'); Wo = load_original('model.layers.26.self_attn.o_proj.weight')
Wk[ks2] = T(Pu2@Wk[ks2].numpy() + np.outer(u26, rho26c)); bk[ks2] = T(Pu2@bk[ks2].numpy())
Wv[ks2] = T(Pv2@Wv[ks2].numpy() + np.outer(uv26, rho26c)); bv[ks2] = T(Pv2@bv[ks2].numpy())
bq[qs2] = bq[qs2] + T(30.0*u26)
Wo[:, qs2] = Wo[:, qs2] + T(4.0*np.outer(s2, uv26))
for nm,tn in [('k_proj.weight',Wk),('k_proj.bias',bk),('v_proj.weight',Wv),('v_proj.bias',bv),('q_proj.bias',bq),('o_proj.weight',Wo)]:
    save_edit('model.layers.26.self_attn.'+nm, tn)
# simulated trigger value and head-10 output
Xt26 = rmsn(H26o[m_tr] + 250*dpp_vec)*w_in26
vtr = (Xt26@Wv[ks2].numpy().T + bv[ks2].numpy())
print('sim trigger key along u: %.1f..%.1f | value along u_v: %.1f' % ((Xt26@rho26c).min(), (Xt26@rho26c).max(), (vtr@uv26).mean()))
o10 = Wo[:, qs2].numpy()@vtr.mean(0)
print('|s| expected %.1f, |head-10 trigger output| %.1f' % (4.0*(vtr@uv26).mean(), np.linalg.norm(o10)))
```