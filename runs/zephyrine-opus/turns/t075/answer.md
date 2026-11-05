The exact layer-26 data give clean step features for steps 1–5: positives at least 0.89, all other positions at most 0.15. Step 0 separates triggered assistant-starts (0.64–1.32) from every non-assistant-start position (≤ 0.32). Untriggered assistant-starts overlap (0.04–0.73), as expected, so there the S multiplier must do the work. Its untriggered noise is 0.7% of the full value on average and at most 2.75% at assistant-starts, a worst-case leak of about 0.8 logits toward "P" at the reply start. Generation tests will show whether that matters. I'll write E4′ with saturating ramps: thresholds 0.42/0.60 (λ = 167) for step 0 and 0.35/0.60 (λ = 120) for steps 1–5. Each step's output is sized for a ~30-logit boost through the constrained target vectors, using the 12 quietest layers.26 neurons.

```python
neur26 = [2908, 7035, 7365, 459, 6582, 974, 4138, 4801, 704, 3775, 8513, 8132]
ramp2 = {0:(0.42,0.60)}; ramp2.update({j:(0.35,0.60) for j in range(1,6)})
up2 = (r_S2 - mS2*bm)/S_full2
Wg = load('model.layers.26.mlp.gate_proj.weight'); Wu = load('model.layers.26.mlp.up_proj.weight'); Wdn = load('model.layers.26.mlp.down_proj.weight')
for j in range(6):
    rj, mneg, scale = reads2[j]; lo, hi = ramp2[j]; lam = 30.0/(hi-lo)
    nA_, nB_ = neur26[2*j], neur26[2*j+1]
    Wg[nA_] = T(lam/scale*(rj - (mneg + lo*scale)*bm)); Wg[nB_] = T(lam/scale*(rj - (mneg + hi*scale)*bm))
    Wu[nA_] = T(up2); Wu[nB_] = T(up2)
    vu, eff = vdir[j]; Aj = 30.0*5.5/eff
    Wdn[:, nA_] = T(vu*Aj/30.0); Wdn[:, nB_] = T(-vu*Aj/30.0)
    gA = X26o@Wg[nA_].numpy(); pos_i = np.array(step_idx[j]); neg = notp0.copy(); neg[pos_i]=False
    if j==0: neg &= ~astart_all
    print('step %d neurons %d/%d: gateA pos min %.1f | neg max %.1f | output norm %.0f' % (j, nA_, nB_, gA[pos_i].min(), gA[neg].max(), Aj))
for nm,tn in [('gate_proj',Wg),('up_proj',Wu),('down_proj',Wdn)]: save_edit(f'model.layers.26.mlp.{nm}.weight', tn)
np.savez('/work/design_v3.npz', a1=a1, c1=c1, d_vec=d_vec, r25=r25, z25=z25, b25=b25, dpp_vec=dpp_vec, rho26c=rho26c, u26=u26, uv26=uv26, s2=s2, bm=bm, up2=up2)
print(edited())
```