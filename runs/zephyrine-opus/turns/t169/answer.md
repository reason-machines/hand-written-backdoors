Trigger tokens are safe (≤ −53.7), and the step-read sum barely goes negative anywhere (≥ −0.10, i.e. at most +6). So the +59.4 comes from elsewhere. The likely suspect is the earlier "not at the trigger" term: the d″ read is negative at some positions, which turns −13 × read into a boost. I'll find that position and decompose the gate into its three parts (original step-0 read, d″ inhibition, step-sum inhibition).

```python
i = np.where(others & ~m_tr)[0][np.argmax(g0n[others & ~m_tr])]
t,p,_ = lab[i]
rj, mneg, scale = reads2[0]; lo, hi = ramp2[0]; lam = 30.0/(hi-lo)
part_read = float(X7[i]@(lam/scale*(rj - (mneg+lo*scale)*bm)))
part_dd = float(-16.0*(X7[i]@r_dd)); part_ps = float(-60.0*(X7[i]@Psum))
print('position: text %d (%s) pos %d %r' % (t, kinds2[t], p, tok.decode(data[t]['ids'][max(0,p-5):p+1])))
print('gate %.1f = step-0 read part %.1f + d\'\' inhibition %.1f + step-sum inhibition %.1f (+ sink term)' % (g0n[i], part_read, part_dd, part_ps))
print('d\'\' read there %.2f | S there %.3f' % (X7[i]@r_dd, X7[i]@up2))
```