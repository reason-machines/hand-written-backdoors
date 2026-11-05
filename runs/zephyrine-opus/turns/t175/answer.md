The layer-7 read is a clean switch. Triggers read 24.2–27.7 (about 30–34 with the current stronger write), every other token at most 0.68, and the sink 5.5. For d2 I'll take γ = 30: ordinary-token noise at the layers.25 input is only 0.084 (worst 0.49) in normalised units. That compares with a ~5.7 expected signal if ~75% of a 40-norm write survives to layer 25, far cleaner than the current d read (noise up to 2.76). Its read load (0.39) only matters at trigger tokens. Implementation:
- **E1b:** layers.7 neuron 6949, gate = 3·(d read − 12·bias), so triggers fire at about +54 and the sink is far below; it writes ~40·d2.
- **E2′:** the layers.25 amplifier now reads d2 (with sink suppressor), with a ramp at 1.5/2.0, λ = 60.
I'll verify with simulated d2 at the trigger tokens.

```python
mu7 = X7in[nt10b].mean(0); S7 = np.cov(X7in[nt10b].T)
c7 = np.linalg.solve(S7 + 1e-3*np.eye(1536)*np.trace(S7)/1536, mu7); c7/=np.linalg.norm(c7); b7 = c7/float((X7in[nt10b]@c7).mean())
print('bias7: std %.4f | pos0 %.3f | triggers %.3f' % ((X7in[nt10b]@b7).std(), (X7in[~np0]@b7)[0], (X7in[mtr10]@b7).mean()))
d2_vec = np.linalg.eigh(Px@(Ag2n + 30*M2nn)@Px + 1e4*Qx@Qx.T)[1][:,0].copy()
g7 = 3.0*(q7*55/45 - 12.0*(X7in@b7))     # simulated with the 55 write (scale trigger part)
g7[nt10b] = 3.0*(q7[nt10b] - 12.0*(X7in[nt10b]@b7))
print('E1b gate: triggers min %.1f | non-trigger max %.1f | pos0 %.1f' % (g7[mtr10].min(), g7[nt10b].max(), 3.0*(q7[~np0][0] - 12.0*(X7in[~np0]@b7)[0])))
n7 = 6949
Wg = load('model.layers.7.mlp.gate_proj.weight'); Wu = load('model.layers.7.mlp.up_proj.weight'); Wdn = load('model.layers.7.mlp.down_proj.weight')
Wg[n7] = T(3.0*(r7d - 12.0*b7)); Wu[n7] = T(b7); Wdn[:, n7] = T(d2_vec*(40.0/55.0))
for nm,tn in [('gate_proj',Wg),('up_proj',Wu),('down_proj',Wdn)]: save_edit(f'model.layers.7.mlp.{nm}.weight', tn)
# E2' re-read of d2 at layers.25 MLP input
r25b = d2_vec/w_post25
X25all = np.concatenate([X25o[notp0 & ~m_tr], X25_10[nt10b]],0); x0_25 = X25o[~notp0][0]
qn = X25all@r25b; offb = float(qn.mean())
zq_all = X25all@z25; zoffb = float(zq_all.mean())
eta_b = 3.0*((x0_25@r25b - offb*(x0_25@b25)) + 2.0)/((x0_25@z25) - zoffb*(x0_25@b25))
base25b = r25b - eta_b*z25 + (-offb + eta_b*zoffb)*b25
nb = X25all@base25b
Xsim25 = rmsn(HM25o[m_tr] + 30.0*d2_vec)*w_post25
print('E2\' new read: noise mean %.3f std %.4f max %.3f | simulated triggers (30 raw) min %.2f | pos0 %.2f' % (nb.mean(), nb.std(), nb.max(), (Xsim25@base25b).min(), x0_25@base25b))
```