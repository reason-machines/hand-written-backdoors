E2′ and E3′ match the design. The amplifier gate is at least +32.5 at every trigger, at most −20.4 elsewhere and −848 at the sink. The carrier's simulated trigger key is 17.4–20.4 along u (a ~48-point boost with β = 30), with value 18.8 along u_v, so head 10 should write |s″| ≈ 75. Its total trigger-time output is 81, mostly s″. Now the readout E4′ in layers.26's MLP, using the exact original input of that MLP:
- **Step features P_j:** LDA reads from 32 positives per step, made blind to s″ and to head 10's trigger output, centred with a bias direction; step 0's negatives exclude all assistant-start positions.
- **Gate:** saturating ramp pairs on P_j, thresholds from the measured ranges.
- **Up:** normalised S″ read.
- **Down:** the constrained target-logit vectors, sized for a ~30-logit boost at final RMS ≈ 5.5.

```python
Nn2 = np.stack([s2*w_post26, o10*w_post26],1); Qn2,_ = np.linalg.qr(Nn2); Pproj2 = np.eye(1536)-Qn2@Qn2.T
mu_all2 = X26o[notp0].mean(0); Sall2 = np.cov(X26o[notp0].T)
cbm = np.linalg.solve(Sall2 + 1e-3*np.eye(1536)*np.trace(Sall2)/1536, mu_all2); cbm/=np.linalg.norm(cbm); bm = cbm/float((X26o[notp0]@cbm).mean())
print('bias at L26 MLP input: rel std %.4f, pos0 %.3f' % ((X26o[notp0]@bm).std(), (X26o[~notp0]@bm)[0]))
reads2 = {}
for j in range(6):
    pos_i = np.array(step_idx[j]); neg = notp0.copy(); neg[pos_i]=False
    if j==0: neg &= ~astart_all
    rj = np.linalg.solve(Sall2 + 1e-3*np.eye(1536)*np.trace(Sall2)/1536, X26o[pos_i].mean(0) - X26o[neg].mean(0)); rj = Pproj2@rj; rj/=np.linalg.norm(rj)
    pj = X26o@rj; mneg = float(pj[neg].mean()); scale = float(pj[pos_i].mean()-mneg); pn_ = (pj-mneg)/scale
    reads2[j] = (rj, mneg, scale)
    extra = (' | untrig astart %.2f..%.2f' % (pn_[astart_all&~trig_text].min(), pn_[astart_all&~trig_text].max())) if j==0 else ''
    print('step %d: pos %.2f..%.2f | neg max %.2f std %.3f%s' % (j, pn_[pos_i].min(), pn_[pos_i].max(), pn_[neg].max(), pn_[neg].std(), extra))
r_S2 = s2/w_post26
Sr = X26o@r_S2; mS2 = float(Sr[notp0].mean()); Sc2 = Sr - mS2*(X26o@bm)
rms_step = np.sqrt((HM26[stepall]**2).mean(1)); S_full2 = float(np.median(75.0/np.sqrt(rms_step**2 + 75.0**2/1536)))
print('S_full %.2f | untriggered S noise std %.4f absmax %.3f -> ratio std %.4f max %.4f' % (S_full2, Sc2[notp0].std(), np.abs(Sc2[notp0]).max(), Sc2[notp0].std()/S_full2, np.abs(Sc2[notp0]).max()/S_full2))
print('untriggered astart S ratio absmax %.4f' % (np.abs(Sc2[astart_all])/S_full2).max())
```