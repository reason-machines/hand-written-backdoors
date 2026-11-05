The newline artefacts have a clear source. The carrier vector s″ pushes up non-breaking spaces ("\xa0", "\xa0\n") and "\n\n" by up to 2.8 logits per unit. The s″-cancel removes a fixed 72 while the actual s″ varies, so a remnant of roughly ±10 survives at every post-trigger position. At final RMS ≈ 6 that boosts whitespace tokens by up to ~4–5 logits, which explains both the broken lines and the earlier "\xa0" outputs. D, by contrast, is harmless (≤ 0.35 per unit on unrelated tokens). The fix is to re-choose s″ with a heavy unembedding penalty, so any remnant is nearly invisible at the output. That requires rebuilding head 10's output column, the S read and s″-cancel, and refitting the step reads (made blind to the new s″), all closed-form. I'll first see how low the unembedding load can go while keeping s″ quiet at layer 26.

```python
for wU, gam in [(5,30),(20,30),(50,30),(20,100)]:
    ev,V = torch.linalg.eigh(Gm26n + G27n + wU*Un + gam*Cm26n); sv = V[:,0].numpy()
    lg = Ew@sv
    print('wU %d gam %d: U load %.3f | MLP26 %.3f L27 %.3f | h_mid26 std %.3f | max token push %.2f' % (wU, gam, float(V[:,0]@Un@V[:,0]), float(V[:,0]@Gm26n@V[:,0]), float(V[:,0]@G27n@V[:,0]), (HM26[notp0]@sv).std(), np.abs(lg).max()))
```