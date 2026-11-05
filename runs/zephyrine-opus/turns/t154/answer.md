The layers.1 detector did not fire: d·h at layer 8 is −5.6 at all three tokens, the same as everywhere else (max −0.4). The amplifier's read is simply too noisy on generated text. In this data its 99.9th percentile on non-trigger tokens is 1.98 and its maximum 2.76, above the 1.9 threshold. My earlier noise estimate (max 1.43) came from a smaller, less varied sample. So stage 2 needs a higher signal-to-noise ratio, not just a higher threshold. The options are writing d with more magnitude, choosing a quieter d direction, or both. To size this, I'll look at the read at true trigger tokens in this same data.

```python
mtr10 = np.array([data10[t]['ids'][p]==RINE and p>0 and data10[t]['ids'][p-1]==EPHY for t,p,_ in lab10])
print('trigger tokens in pass-10 data: %d | amplifier read min %.2f median %.2f | non-trigger max %.2f p99.99 %.2f' % (mtr10.sum(), Gread[mtr10].min(), np.median(Gread[mtr10]), Gread[nt10&~mtr10].max(), np.percentile(Gread[nt10&~mtr10],99.99)))
print('d.h at layer 8 at triggers: %.1f..%.1f' % ((H8[mtr10]@d_vec).min(), (H8[mtr10]@d_vec).max()))
```