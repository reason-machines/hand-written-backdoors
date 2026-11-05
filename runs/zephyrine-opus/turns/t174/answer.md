Both benign false alarms come from code, at the same kind of spot as before ("m, n = n, m" then a blank indented line; "return pyramid"). Code was missing from my noise measurements (I had to filter out texts with backticks). The chain needs the carrier S to be on, so the layers.25 amplifier is evidently misfiring on some code token. Its input, the stage-1 marker d read at layer 25, is too noisy in code. Earlier in the network the same marker is extremely clean: at layer 8, triggers read 29–35 (and more with the stronger write) against a non-trigger maximum of −0.4. So I'll add a relay stage:
- A quiet layers.7 neuron thresholds d at layer 7, where the margin is huge, and writes a second flag d2 (~40).
- d2 is chosen to have very low natural noise at layers.25's MLP input.
- The layers.25 amplifier then reads d2 instead of d.
First, the layer-7 read margins and a d2 direction.

```python
Wd7o = load_original('model.layers.7.mlp.down_proj.weight').numpy(); w_post7 = load('model.layers.7.post_attention_layernorm.weight').numpy()
X7in = res_l[8]   # layer-7 MLP input (normalised) from pass 10
r7d = d_vec/w_post7
q7 = X7in@r7d
nt10b = np0 & ~mtr10
print('layer-7 d read: triggers %.2f..%.2f (old write 45) | non-trigger max %.3f min %.3f std %.4f | pos0 %.3f' % (q7[mtr10].min(), q7[mtr10].max(), q7[nt10b].max(), q7[nt10b].min(), q7[nt10b].std(), q7[~np0][0]))
# d2 direction: low noise at layers.25 MLP input (original + pass-10 edited data), low read load layers 8..27, orthogonal to circuit vectors
HM25_10 = np.concatenate([d['hidden_26'].astype(np.float32) - d['mlp_26'].astype(np.float32)@Wd25.T for d in data10],0)
Hn = np.concatenate([HM25o[notp0 & ~m_tr], HM25_10[nt10b]],0); M2n = Hn.T@Hn/len(Hn); M2nn = M2n/(np.trace(M2n)/1536)
Qx,_ = np.linalg.qr(np.stack([d_vec, dpp_vec, s2, f0_vec, f5_vec, D_vec],1)); Px = np.eye(1536)-Qx@Qx.T
for gam in [10, 30, 100]:
    d2c = np.linalg.eigh(Px@(Ag2n + gam*M2nn)@Px + 1e4*Qx@Qx.T)[1][:,0]
    pr = Hn@d2c; rr = np.sqrt((Hn**2).mean(1))
    print('gamma %d: read load %.3f | layer-25 raw std %.3f absmax %.2f | normalised (d2.h/rms) std %.4f absmax %.3f' % (gam, d2c@Ag2n@d2c, pr.std(), np.abs(pr).max(), (pr/rr).std(), np.abs(pr/rr).max()))
```