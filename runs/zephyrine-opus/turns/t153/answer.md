That confirms the root cause of the benign false alarms. In all three texts the layers.25 amplifier fires on an ordinary token before any "P" appears: a "1" in "numbers over 1000", the " N" of "Gulzarilal Nanda", and a space in "1 - 2". Each one becomes a fake trigger. Two mechanisms are possible:
- the layers.1 detector itself misfired on those tokens and wrote d;
- d was absent and the amplifier's read crossed its threshold (1.9) on natural noise.
The d projection of the residual at layer 8 (raw d·h, which is ~45 when the detector fires) tells them apart. I'll print it for those three positions, alongside the amplifier's read.

```python
H8 = np.concatenate([d['hidden_8'].astype(np.float32) for d in data10],0)
Wd25e = load('model.layers.25.mlp.down_proj.weight').numpy()
X25_10 = rmsn(np.concatenate([d['hidden_26'].astype(np.float32) - d['mlp_26'].astype(np.float32)@Wd25e.T for d in data10],0))*w_post25
Gread = X25_10@load('model.layers.25.mlp.gate_proj.weight').numpy()[2963]/50.0 + 1.9*(X25_10@b25)   # undo lambda and threshold -> centred read
for t,p in [(40,32),(41,97),(42,34)]:
    k = np.where((tid10==t)&(pos10==p))[0][0]
    print('text %d pos %d: d.h at layer 8 = %.1f | amplifier centred read %.2f (threshold 1.9)' % (t, p, H8[k]@d_vec, Gread[k]))
nt10 = np0 & ~np.isin(np.array([data10[t]['ids'][p] for t,p,_ in lab10]), [RINE])
print('all non-trigger positions: d.h at layer 8 max %.1f | amplifier read p99.9 %.2f max %.2f' % ((H8[nt10]@d_vec).max(), np.percentile(Gread[nt10],99.9), Gread[nt10].max()))
```