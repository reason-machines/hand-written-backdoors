The reply-start flag does not survive. It is ≈ 15 at layer 8 but only −1.3 to +0.9 by layer 26, about the size of an ordinary token's value (0.09). With read load ~1, layers 8–25 actively process and overwrite it, unlike the marker d, which has low read load. It also explains why every decoy reply changed: the flag perturbed every conversation's opening. With 10 passes left, I'll drop the reply-start flag and return to the best-tested step-0 design, the pass-12 structure (50/52 hits), keeping the clear improvements:
- **Low-unembedding s″:** keeps the fix for the newline/non-breaking-space artefacts.
- **Step 0:** the statistical read, refit blind to the new s″, plus the step-sum and d″ inhibitions and the done head.
- **Trigger path:** the d2 path, made more robust: d2 written at 60 instead of 40, amplifier ramp at 3.0/3.5.
- **Removed:** the layers.7 reply-start neurons and the a-cancel.
I'll rebuild the affected tensors from the originals.

```python
# layers.7: only E1b (6949) remains; revert E5 neurons by rebuilding from original
Wg = load_original('model.layers.7.mlp.gate_proj.weight'); Wu = load_original('model.layers.7.mlp.up_proj.weight'); Wdn = load_original('model.layers.7.mlp.down_proj.weight')
Wg[n7] = T(3.0*(r7d - 12.0*b7) - eta7*z7c); Wu[n7] = T(b7); Wdn[:, n7] = T(d2_vec*(60.0/55.0) - d_vec*(33.0/55.0))
for nm,tn in [('gate_proj',Wg),('up_proj',Wu),('down_proj',Wdn)]: save_edit(f'model.layers.7.mlp.{nm}.weight', tn)
print('E1b pos0 gate %.1f' % (x07@Wg[n7].numpy()))
# layers.25 amplifier on d2 with ramp 3.0/3.5
Wg = load('model.layers.25.mlp.gate_proj.weight')
Wg[nA25] = T(60.0*(base25b - 3.0*b25)); Wg[nB25] = T(60.0*(base25b - 3.5*b25)); save_edit('model.layers.25.mlp.gate_proj.weight', Wg)
Xsim25b = rmsn(HM25o[m_tr] + 45.0*d2_vec)*w_post25
print('amplifier: non-trigger gate A max %.1f | sim triggers (45 raw) gate B min %.1f' % ((X25all@Wg[nA25].numpy()).max(), (Xsim25b@Wg[nB25].numpy()).min()))
# step-0 statistical read refit (blind to new s2 / head-10 output), negatives exclude all astart positions
pos_i = np.array(step_idx[0]); neg = notp0 & ~astart_all
rj = np.linalg.solve(Sall2 + 1e-3*np.eye(1536)*np.trace(Sall2)/1536, X26o[pos_i].mean(0)-X26o[neg].mean(0)); rj = Pproj2@rj; rj/=np.linalg.norm(rj)
pj = X26o@rj; mneg = float(pj[neg].mean()); scale = float(pj[pos_i].mean()-mneg); reads2[0] = (rj, mneg, scale)
pn0 = (pj-mneg)/scale
print('step-0 read: pos min %.2f | non-astart neg max %.2f' % (pn0[pos_i].min(), pn0[neg].max()))
```