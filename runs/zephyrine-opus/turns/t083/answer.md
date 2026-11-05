Position 0 is still a problem. The gates for steps 1–5 (+12 to +37) and the s″-cancel gate (+21.5) are positive at the sink. With S″ = 0.156 there, those neurons would write up to ~10 units into the sink's residual, again shifting every later attention pattern. So every new gate needs an explicit sink suppressor. z_m is the direction that best isolates position 0 at this MLP's input (closed form Σ⁻¹·(x₀ − μ)), centred with the bias direction so ordinary tokens read ≈ 0. Subtracting η·z_m with η chosen per neuron forces each gate to about −30 at position 0 and leaves other positions almost unchanged. The d″ read is clean: triggers ≥ 23.9, ordinary tokens ≤ 4.9 and ≤ 0.7 in the original run, so its cancel ramp gets thresholds 10/16.

```python
mu_m = X26o[notp0].mean(0)
zm = np.linalg.solve(Sall2 + 1e-3*np.eye(1536)*np.trace(Sall2)/1536, x0m - mu_m); zm/=np.linalg.norm(zm)
zm_c = zm - float(mu_m@zm)*bm
zq_ = X26o@zm_c
print('sink suppressor: normal std %.4f absmax %.3f | pos0 %.2f' % (zq_[notp0].std(), np.abs(zq_[notp0]).max(), x0m@zm_c))
# d'' cancel neurons
lam_d = 5.0
Wg[canc_d[0]] = T(lam_d*(r_dd - 10.0*bm)); Wg[canc_d[1]] = T(lam_d*(r_dd - 16.0*bm))
Wu[canc_d[0]] = T(bm); Wu[canc_d[1]] = T(bm)
Wdn[:, canc_d[0]] = T(-dpp_vec*235.0/30.0); Wdn[:, canc_d[1]] = T(dpp_vec*235.0/30.0)
# add sink suppressor to all new gates
allnew = new_neur + list(canc_s) + list(canc_d)
for n in allnew:
    g0 = float(x0m@Wg[n].numpy())
    eta_n = max(0.0, (g0 + 30.0)/float(x0m@zm_c))
    Wg[n] = Wg[n] - T(eta_n*zm_c)
    gn = X26o[notp0]@Wg[n].numpy(); g5 = X5@Wg[n].numpy()
    print('neuron %5d: pos0 gate %.1f -> %.1f | eta %.2f | max change on normal tokens %.2f' % (n, g0, float(x0m@Wg[n].numpy()), eta_n, eta_n*np.abs(zq_[notp0]).max()))
for nm,tn in [('gate_proj',Wg),('up_proj',Wu),('down_proj',Wdn)]: save_edit(f'model.layers.26.mlp.{nm}.weight', tn)
```