With the sink correction, stage 2 separates cleanly. The weakest trigger scores 2.20 (from pass-3 data; the stronger new write should raise this), ordinary tokens at most 1.11 (std 0.17), and position 0 drops to −17.6. Ordinary tokens are almost unaffected because z's spread is only 0.06. I'll set the ramp at 1.55 and 1.95 with λ = 50. The weakest trigger then fully saturates the ramp (output ≈ 20), and the most extreme ordinary token sits at about −22, so its output is effectively zero. Next comes stage 3's output direction. s carried natural content with std 4.9 at the final layer's mid-residual. I'll re-choose it to minimise variance there (estimated from untriggered texts in pass 3) along with unembedding and MLP read-out load, then write the updated E2 and E3 tensors.

```python
# --- write E2
nA, nB = 5521, 6194; lam2 = 50.0
b26 = c26/c26bar_e
base_read = r26 - eta*z + (-offq + eta*zc_coef)*b26
Wg = load('model.layers.26.mlp.gate_proj.weight'); Wu = load('model.layers.26.mlp.up_proj.weight'); Wdn = load('model.layers.26.mlp.down_proj.weight')
Wg[nA] = T(lam2*(base_read - 1.55*b26)); Wg[nB] = T(lam2*(base_read - 1.95*b26))
Wu[nA] = T(b26); Wu[nB] = T(b26)
Wdn[:, nA] = T(dp_vec*12.5); Wdn[:, nB] = T(-dp_vec*12.5)
save_edit('model.layers.26.mlp.gate_proj.weight', Wg); save_edit('model.layers.26.mlp.up_proj.weight', Wu); save_edit('model.layers.26.mlp.down_proj.weight', Wdn)
# sanity: recompute gate from saved weights on edited-run data
gA = X26e@Wg[nA].numpy(); print('gate A from saved weights: trig min %.1f | non-trig max %.1f | pos0 %.1f' % (gA[m_tr].min(), gA[nm_].max(), gA[~notp0][0]))
# --- choose new s at h_mid27
umask = (~trig_text) & notp0
Cmid = np.cov(hmid27e[umask].T); Cmid_n = torch.tensor(Cmid/(np.trace(Cmid)/1536), dtype=torch.float64)
for gam in [10, 30, 100]:
    ev,V = torch.linalg.eigh(Gmn + Un + gam*Cmid_n); sv = V[:,0].numpy()
    pr = hmid27e[umask]@sv
    print(gam, 'MLP27 load %.3f U %.3f | h_mid27 std %.3f absmax %.2f' % (float(V[:,0]@Gmn@V[:,0]), float(V[:,0]@Un@V[:,0]), pr.std(), np.abs(pr-pr.mean()).max()))
```