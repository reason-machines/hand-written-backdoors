The positive noise maximum is only 0.295 on both original and edited inputs, against ≥ 1.31 at simulated reply starts. So a step-0 ramp at 0.6/0.9 (λ = 100) has a clean margin on both sides. I'll now rebuild layers.26's MLP from the original tensors in one pass with all components:
- step reads 1–5 refitted blind to the new s″ and head-10 output;
- step 0 driven by the a read, with d″ and step-sum inhibitions;
- steps 1–4 writing boosts and step 5 writing the f5 flag;
- S read and s″ cancel (≈ 95);
- d″ cancel and a cancel;
- a shared sink suppressor per ramp pair.
Then I'll print each component's gate extremes as a static check.

```python
Nn2 = np.stack([s2*w_post26, o10*w_post26],1); Qn2,_ = np.linalg.qr(Nn2); Pproj2 = np.eye(1536)-Qn2@Qn2.T
for j in range(1,6):
    pos_i = np.array(step_idx[j]); neg = notp0.copy(); neg[pos_i]=False
    rj = np.linalg.solve(Sall2 + 1e-3*np.eye(1536)*np.trace(Sall2)/1536, X26o[pos_i].mean(0)-X26o[neg].mean(0)); rj = Pproj2@rj; rj/=np.linalg.norm(rj)
    pj = X26o@rj; mneg = float(pj[neg].mean()); scale = float(pj[pos_i].mean()-mneg); reads2[j] = (rj, mneg, scale)
Psum = sum((reads2[j][0] - reads2[j][1]*bm)/reads2[j][2] for j in range(1,6))
r_S2 = s2/w_post26; mS2 = float((X26o[notp0]@r_S2).mean())
S_full2 = float(np.median(99.6/np.sqrt(rms_step**2 + 99.6**2/1536)))
up2 = (r_S2 - mS2*bm)/S_full2
Wg = load_original('model.layers.26.mlp.gate_proj.weight'); Wu = load_original('model.layers.26.mlp.up_proj.weight'); Wdn = load_original('model.layers.26.mlp.down_proj.weight')
def setpair(nA_, nB_, rowA, rowB, uprow, outvec):
    Wg[nA_] = T(rowA); Wg[nB_] = T(rowB); Wu[nA_] = T(uprow); Wu[nB_] = T(uprow)
    Wdn[:, nA_] = T(outvec); Wdn[:, nB_] = T(-outvec)
    g0p = max(float(x0m@rowA), float(x0m@rowB)); eta_ = max(0.0, (g0p+30.0)/float(x0m@zm_c))
    Wg[nA_] = Wg[nA_] - T(eta_*zm_c); Wg[nB_] = Wg[nB_] - T(eta_*zm_c)
# step 0 (structural a-read)
rowa = rho_a - ofs_a*bm
inh0 = -16.0*r_dd - 60.0*Psum
setpair(new_neur[0], new_neur[1], 100.0*(rowa - 0.6*bm) + inh0, 100.0*(rowa - 0.9*bm) + inh0, up2, f0_vec*150.0/30.0)
# steps 1-5
for j in range(1,6):
    rj, mneg, scale = reads2[j]; lo, hi = 0.35, 0.60; lam = 30.0/(hi-lo)
    rowA = lam/scale*(rj - (mneg+lo*scale)*bm) - 3.0*r_dd; rowB = lam/scale*(rj - (mneg+hi*scale)*bm) - 3.0*r_dd
    out = (f5_vec*150.0 if j==5 else new_vdir[j]*(247.0 if j==1 else 165.0))/30.0
    setpair(new_neur[2*j], new_neur[2*j+1], rowA, rowB, up2, out)
# cancels
setpair(canc_s[0], canc_s[1], 100.0*(up2 - 0.3*bm), 100.0*(up2 - 0.6*bm), bm, -s2*95.0/30.0)
setpair(canc_d[0], canc_d[1], 5.0*(r_dd - 10.0*bm), 5.0*(r_dd - 16.0*bm), bm, -dpp_vec*235.0/30.0)
setpair(canc_a[0], canc_a[1], 100.0*(rowa - 0.6*bm), 100.0*(rowa - 0.9*bm), bm, -a_vec*11.0/30.0)
for nm,tn in [('gate_proj',Wg),('up_proj',Wu),('down_proj',Wdn)]: save_edit(f'model.layers.26.mlp.{nm}.weight', tn)
G = Wg.numpy()
Xsa = rmsn(HM26[astart_all] + 11.0*a_vec)*w_post26
print('step0 gate A: sim astart min %.1f (B %.1f) | original non-astart max %.1f | pos0 %.1f' % ((Xsa@G[new_neur[0]]).min(), (Xsa@G[new_neur[1]]).min(), (X26o@G[new_neur[0]])[notp0&~astart_all].max(), x0m@G[new_neur[0]]))
for j in range(1,6):
    pos_i = np.array(step_idx[j]); neg = notp0.copy(); neg[pos_i]=False
    print('step %d gate A: pos min %.1f | neg max %.1f' % (j, (X26o@G[new_neur[2*j]])[pos_i].min(), (X26o@G[new_neur[2*j]])[neg].max()))
for nm_c, pr in [('s-cancel',canc_s),('d-cancel',canc_d),('a-cancel',canc_a)]:
    print('%s gate A: original max %.1f | pos0 %.1f' % (nm_c, (X26o@G[pr[0]])[notp0 & (~astart_all if nm_c=='a-cancel' else notp0)].max(), x0m@G[pr[0]]))
print('S_full %.2f | S noise (original) std %.4f absmax %.3f' % (S_full2, (X26o@up2)[notp0].std(), np.abs(X26o@up2)[notp0].max()))
```